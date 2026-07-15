"""
Graph Heterogeneous Collaborative Filtering (GHCF) model for eHMI recommendation.
PyTorch implementation adapted from the original TensorFlow version.
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
try:
    from HeteroGNN import (
        HeteroGATLayer,
        HeteroGCNLayer,
        HeteroGraphSAGELayer,
        MultiRelationEmbedding,
    )
except Exception:
    from models.HeteroGNN import (
        HeteroGATLayer,
        HeteroGCNLayer,
        HeteroGraphSAGELayer,
        MultiRelationEmbedding,
    )


class GHCF(nn.Module):
    """
    Graph Heterogeneous Collaborative Filtering model for multi-relational recommendation.

    This model handles heterogeneous user-item interactions through:
    1. Embedding propagation layers for both nodes and relations
    2. Multi-task prediction module for different interaction types
    3. Efficient non-sampling learning optimization
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_user_features: int,
        n_item_features: int,
        interaction_types: List[str],
        embedding_dim: int = 64,
        layer_sizes: List[int] = [64, 32],
        dropout_rates: List[float] = [0.1, 0.1],
        gnn_type: str = "gcn",
        n_heads: int = 4,
        device: str = "cpu",
        bound_bfi_output: bool = True,
    ):
        """
        Initialize GHCF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            n_user_features: Number of user features
            n_item_features: Number of item features
            interaction_types: List of interaction type names
            embedding_dim: Embedding dimension
            layer_sizes: Sizes of GNN layers
            dropout_rates: Dropout rates for each layer
            gnn_type: Type of GNN layer ('gcn', 'sage', 'gat')
            n_heads: Number of attention heads (only used for 'gat')
            device: Device to run the model on
            bound_bfi_output: Squash the BFI perception head through a sigmoid so
                it stays in [0, 1] (the MinMax target / causal-input domain)
        """
        super(GHCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.interaction_types = interaction_types
        self.embedding_dim = embedding_dim
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.gnn_type = gnn_type
        self.device = device
        self.bound_bfi_output = bound_bfi_output

        # Feature projection layers
        self.user_feature_proj = nn.Linear(n_user_features, embedding_dim)
        self.item_feature_proj = nn.Linear(n_item_features, embedding_dim)

        # Initial embeddings (learnable)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Import config to get interaction type dimensions
        try:
            from configs.config import INTERACTION_TYPES
        except ImportError:
            # Fallback to hardcoded values if config not available
            INTERACTION_TYPES = {
                "RATING": {"columns": ["Rating score"]},
                "BFI_PERCEPTION": {"columns": ["BFI-O", "BFI-C", "BFI-E", "BFI-A", "BFI-N"]},
                "VOICE_QUALITY": {
                    "columns": ["MOS-X2 Q1", "MOS-X2 Q2", "MOS-X2 Q3", "MOS-X2 Q4"]
                },
                "SUBJECTIVE_FEELING": {"columns": ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5"]},
            }

        # Calculate total number of relations (sum of all components across interaction types)
        self.interaction_components = {}  # Maps interaction_type -> list of component names
        self.component_to_relation_idx = (
            {}
        )  # Maps (interaction_type, component_idx) -> global relation index
        total_relations = 0

        for interaction_type in interaction_types:
            if interaction_type in INTERACTION_TYPES:
                components = INTERACTION_TYPES[interaction_type]["columns"]
                self.interaction_components[interaction_type] = components

                for comp_idx, component in enumerate(components):
                    self.component_to_relation_idx[(interaction_type, comp_idx)] = total_relations
                    total_relations += 1
            else:
                # Default to single component for unknown interaction types
                self.interaction_components[interaction_type] = ["default"]
                self.component_to_relation_idx[(interaction_type, 0)] = total_relations
                total_relations += 1

        self.n_relations = total_relations

        # Relation embeddings
        self.relation_embedding = MultiRelationEmbedding(self.n_relations, embedding_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        layer_dims = [embedding_dim] + layer_sizes

        for i in range(self.n_layers):
            if gnn_type == "gcn":
                layer = HeteroGCNLayer(
                    layer_dims[i],
                    layer_dims[i + 1],
                    self.n_relations,
                    dropout_rates[i] if i < len(dropout_rates) else 0.1,
                )
            elif gnn_type == "sage":
                layer = HeteroGraphSAGELayer(
                    layer_dims[i],
                    layer_dims[i + 1],
                    self.n_relations,
                    dropout_rates[i] if i < len(dropout_rates) else 0.1,
                )
            elif gnn_type == "gat":
                layer = HeteroGATLayer(
                    layer_dims[i],
                    layer_dims[i + 1],
                    self.n_relations,
                    n_heads=n_heads,
                    dropout=dropout_rates[i] if i < len(dropout_rates) else 0.1,
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.gnn_layers.append(layer)

        # Prediction layers for each interaction type (but predict all components together)
        final_dim = layer_sizes[-1] if layer_sizes else embedding_dim
        self.prediction_layers = nn.ModuleDict()

        for interaction_type in interaction_types:
            # Get output dimension from config
            if interaction_type in INTERACTION_TYPES:
                output_dim = len(INTERACTION_TYPES[interaction_type]["columns"])
            else:
                # Default to single output for unknown interaction types
                output_dim = 1

            self.prediction_layers[interaction_type] = nn.Sequential(
                nn.Linear(final_dim * 2, final_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(final_dim, output_dim),
            )

        # Initialize weights
        self._init_weights()

        self.debug_mode = False

        # RATING loss configuration (set by the Trainer; defaults preserve the
        # original MSE-only behavior). In "hybrid" mode the RATING loss becomes
        # MSE + bpr_lambda * BPR. BFI/VQ/SF always use MSE.
        self.loss_mode = "mse"
        self.bpr_lambda = 1.0
        self.bpr_num_neg = 1
        self.rating_threshold = 0.0
        # {user_id: set(item_id)} of positively-rated items, for BPR negatives
        self.rating_pos_items: Optional[Dict[int, set]] = None
        # Per-batch breakdown of the RATING loss (detached), for logging the MSE
        # and BPR components separately. Populated only in hybrid mode.
        self.last_rating_components: Dict[str, float] = {}

    def _init_weights(self) -> None:
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.user_feature_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.item_feature_proj.weight, gain=0.1)

        # Initialize biases to zero
        if self.user_feature_proj.bias is not None:
            nn.init.zeros_(self.user_feature_proj.bias)
        if self.item_feature_proj.bias is not None:
            nn.init.zeros_(self.item_feature_proj.bias)

        # Initialize prediction layers
        for layer_dict in self.prediction_layers.values():
            for layer in layer_dict:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def create_node_embeddings(
        self, user_features: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Create initial node embeddings combining learnable and feature-based embeddings.

        Args:
            user_features: User feature matrix [n_users, n_user_features]
            item_features: Item feature matrix [n_items, n_item_features]

        Returns:
            Combined node embeddings [n_users + n_items, embedding_dim]
        """

        # Learnable embeddings
        user_emb = self.user_embedding.weight  # [n_users, embedding_dim]
        item_emb = self.item_embedding.weight  # [n_items, embedding_dim]

        # Feature-based embeddings
        user_feat_emb = self.user_feature_proj(user_features)  # [n_users, embedding_dim]
        item_feat_emb = self.item_feature_proj(item_features)  # [n_items, embedding_dim]

        # Combine learnable and feature embeddings
        user_combined = user_emb + user_feat_emb
        item_combined = item_emb + item_feat_emb

        # Concatenate user and item embeddings
        node_embeddings = torch.cat([user_combined, item_combined], dim=0)

        return node_embeddings

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        adj_matrices: Dict[str, List[torch.sparse.FloatTensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GHCF model.

        Args:
            user_features: User features [n_users, n_user_features]
            item_features: Item features [n_items, n_item_features]
            adj_matrices: Dictionary mapping interaction_type -> list of adjacency matrices for each component

        Returns:
            Tuple of (user_embeddings, item_embeddings, relation_embeddings)
        """
        # Create initial node embeddings
        node_embeddings = self.create_node_embeddings(user_features, item_features)
        expected_total_nodes = self.n_users + self.n_items

        # Prepare adjacency matrices in order and validate shapes
        adj_matrix_list = []
        for interaction_type in self.interaction_types:
            if interaction_type in adj_matrices:
                matrices_for_type = adj_matrices[interaction_type]
                expected_num_components = len(self.interaction_components[interaction_type])

                if len(matrices_for_type) != expected_num_components:
                    raise ValueError(
                        f"Expected {expected_num_components} adjacency matrices for {interaction_type}, "
                        f"got {len(matrices_for_type)}"
                    )

                for comp_idx, adj_matrix in enumerate(matrices_for_type):
                    # Validate adjacency matrix shape
                    expected_shape = (expected_total_nodes, expected_total_nodes)
                    if adj_matrix.shape != expected_shape:
                        component_name = self.interaction_components[interaction_type][comp_idx]
                        raise ValueError(
                            f"Adjacency matrix for {interaction_type}.{component_name} has shape {adj_matrix.shape}, "
                            f"expected {expected_shape} (n_users + n_items = {expected_total_nodes})"
                        )

                    adj_matrix = adj_matrix.coalesce()

                    if adj_matrix.values().isnan().any():
                        raise ValueError(
                            f"Adjacency matrix for {interaction_type}.{self.interaction_components[interaction_type][comp_idx]} contains NaN values"
                        )
                    adj_matrix_list.append(adj_matrix)
            else:
                # Create empty sparse matrices if interaction type not present
                expected_num_components = len(self.interaction_components[interaction_type])
                for _ in range(expected_num_components):
                    adj_matrix_list.append(None)

        # Validate we have the expected number of adjacency matrices
        assert (
            len(adj_matrix_list) == self.n_relations
        ), f"Expected {self.n_relations} adjacency matrices (total components), got {len(adj_matrix_list)}"

        # Store embeddings from each layer
        all_embeddings = [node_embeddings]

        # Base (layer-0) relation embeddings, refined as they propagate through
        # the layers (GHCF threads the relation embeddings through propagation
        # and modulates each relation's messages by them).
        relation_embeddings = self.relation_embedding(torch.tensor([]))
        all_relation_embeddings = [relation_embeddings]

        # Pass through GNN layers
        current_embeddings = node_embeddings
        for layer_idx, gnn_layer in enumerate(self.gnn_layers):
            # Validate embedding shape before passing to GNN layer
            assert (
                current_embeddings.shape[0] == expected_total_nodes
            ), f"Layer {layer_idx}: embedding shape {current_embeddings.shape} has wrong number of nodes"

            current_embeddings, relation_embeddings = gnn_layer(
                current_embeddings, adj_matrix_list, relation_embeddings
            )
            all_embeddings.append(current_embeddings)
            if relation_embeddings is not None:
                all_relation_embeddings.append(relation_embeddings)

            # Validate embedding shape after GNN layer
            assert (
                current_embeddings.shape[0] == expected_total_nodes
            ), f"Layer {layer_idx}: output embedding shape {current_embeddings.shape} has wrong number of nodes"

        # Average embeddings from all layers
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)

        # Split into user and item embeddings
        user_embeddings = final_embeddings[: self.n_users]
        item_embeddings = final_embeddings[self.n_users :]

        # Validate final embedding shapes
        assert user_embeddings.shape == (
            self.n_users,
            final_embeddings.shape[1],
        ), f"User embeddings shape {user_embeddings.shape} is incorrect"
        assert item_embeddings.shape == (
            self.n_items,
            final_embeddings.shape[1],
        ), f"Item embeddings shape {item_embeddings.shape} is incorrect"

        # Relation embeddings: mean across layers (GHCF-style), mirroring the
        # node-embedding layer aggregation above. With a single layer this is
        # the mean of the base and the once-transformed relation embeddings.
        relation_embeddings = torch.stack(all_relation_embeddings, dim=0).mean(dim=0)

        # Validate relation embeddings shape
        assert (
            relation_embeddings.shape[0] == self.n_relations
        ), f"Relation embeddings shape {relation_embeddings.shape} has wrong number of relations"

        return user_embeddings, item_embeddings, relation_embeddings

    def _bound_output(self, interaction_type: str, prediction: torch.Tensor) -> torch.Tensor:
        """Bound the BFI perception head to [0, 1] with a sigmoid.

        BFI targets are MinMax-scaled to [0, 1], and the BFI prediction is the
        input to the causal SEM head (whose rescaled BFI-TIPI difference assumes
        BFI in [0, 1]). Squashing makes that invariant exact. The other heads
        (RATING/VQ/SF are z-scored) live in an unbounded space and are left as-is.
        """
        if interaction_type == "BFI_PERCEPTION" and getattr(self, "bound_bfi_output", False):
            return torch.sigmoid(prediction)
        return prediction

    def predict_interaction(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        interaction_type: str,
        component_idx: int = None,
        user_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict interaction scores for specific user-item pairs.

        Args:
            user_embeddings: User embeddings [n_users, embedding_dim]
            item_embeddings: Item embeddings [n_items, embedding_dim]
            relation_embeddings: Relation embeddings [n_relations, embedding_dim]
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            interaction_type: Type of interaction to predict
            component_idx: Specific component index to predict (if None, predict all components)
            user_features: User features [n_users, n_features]; unused by the
                base model, consumed by causal subclasses

        Returns:
            Predicted interaction scores [batch_size, output_dim] or [batch_size] for single component
        """
        # Validate input shapes
        assert (
            user_embeddings.shape[0] >= user_ids.max() + 1
        ), f"User embeddings shape {user_embeddings.shape} insufficient for user_ids max {user_ids.max()}"
        assert (
            item_embeddings.shape[0] >= item_ids.max() + 1
        ), f"Item embeddings shape {item_embeddings.shape} insufficient for item_ids max {item_ids.max()}"

        # Standard graph-based prediction for all interaction types
        batch_user_emb = user_embeddings[user_ids]  # [batch_size, embedding_dim]
        batch_item_emb = item_embeddings[item_ids]  # [batch_size, embedding_dim]

        if interaction_type not in self.interaction_types:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        if component_idx is not None:
            # Predict for a specific component
            if component_idx >= len(self.interaction_components[interaction_type]):
                raise ValueError(
                    f"Component index {component_idx} out of bounds for {interaction_type} "
                    f"which has {len(self.interaction_components[interaction_type])} components"
                )

            # Get relation embedding for this specific component
            relation_idx = self.component_to_relation_idx[(interaction_type, component_idx)]
            assert (
                relation_idx < relation_embeddings.shape[0]
            ), f"Relation index {relation_idx} out of bounds for relation_embeddings shape {relation_embeddings.shape}"

            relation_emb = relation_embeddings[relation_idx]  # [embedding_dim]

            # Element-wise product with relation embedding
            user_relation = batch_user_emb * relation_emb.unsqueeze(
                0
            )  # [batch_size, embedding_dim]
            item_relation = batch_item_emb * relation_emb.unsqueeze(
                0
            )  # [batch_size, embedding_dim]

            # Concatenate user and item embeddings
            combined_emb = torch.cat(
                [user_relation, item_relation], dim=1
            )  # [batch_size, 2*embedding_dim]

            # Predict single component score
            graph_prediction = self.prediction_layers[interaction_type](combined_emb)
            graph_prediction = self._bound_output(interaction_type, graph_prediction)
            return graph_prediction[:, component_idx : component_idx + 1]  # [batch_size, 1]

        else:
            # Predict for all components of this interaction type
            # Average the relation embeddings for all components of this interaction type
            component_relation_indices = []
            for comp_idx in range(len(self.interaction_components[interaction_type])):
                relation_idx = self.component_to_relation_idx[(interaction_type, comp_idx)]
                component_relation_indices.append(relation_idx)

            # Get average relation embedding for this interaction type
            component_relation_embs = relation_embeddings[
                component_relation_indices
            ]  # [n_components, embedding_dim]
            avg_relation_emb = component_relation_embs.mean(dim=0)  # [embedding_dim]

            # Element-wise product with average relation embedding
            user_relation = batch_user_emb * avg_relation_emb.unsqueeze(
                0
            )  # [batch_size, embedding_dim]
            item_relation = batch_item_emb * avg_relation_emb.unsqueeze(
                0
            )  # [batch_size, embedding_dim]

            # Concatenate user and item embeddings
            combined_emb = torch.cat(
                [user_relation, item_relation], dim=1
            )  # [batch_size, 2*embedding_dim]

            # Predict all component scores using graph model
            graph_prediction = self.prediction_layers[interaction_type](combined_emb)

            return self._bound_output(interaction_type, graph_prediction)

    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        adj_matrices: Dict[str, List[torch.sparse.FloatTensor]],
        interaction_type: str = "RATING",
        component_idx: int = None,
    ) -> torch.Tensor:
        """
        End-to-end prediction for user-item pairs.

        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            user_features: User features [n_users, n_user_features]
            item_features: Item features [n_items, n_item_features]
            adj_matrices: Dictionary mapping interaction_type -> list of adjacency matrices for each component
            interaction_type: Type of interaction to predict
            component_idx: Specific component to predict (if None, predict all components)

        Returns:
            Predicted scores [batch_size] for single-component or [batch_size, output_dim] for multi-component
        """
        # Forward pass to get embeddings
        user_embeddings, item_embeddings, relation_embeddings = self.forward(
            user_features, item_features, adj_matrices
        )

        # Predict interactions
        predictions = self.predict_interaction(
            user_embeddings,
            item_embeddings,
            relation_embeddings,
            user_ids,
            item_ids,
            interaction_type,
            component_idx=component_idx,
            user_features=user_features,
        )

        return predictions

    def get_all_interaction_types(self) -> List[str]:
        """
        Get all interaction types this model can predict.

        Causal subclasses override this to include interaction types that are
        predicted via the causal head rather than the GNN.
        """
        return list(self.interaction_types)

    @staticmethod
    def _group_batch_by_type(
        batch_data: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Group batch entries by interaction type.

        Returns:
            Dictionary mapping interaction_type -> (user_ids, item_ids, values)
        """
        grouped = {}
        for i, interaction_type in enumerate(batch_data["interaction_types"]):
            if interaction_type not in grouped:
                grouped[interaction_type] = {"users": [], "items": [], "values": []}
            grouped[interaction_type]["users"].append(batch_data["users"][i])
            grouped[interaction_type]["items"].append(batch_data["items"][i])
            grouped[interaction_type]["values"].append(batch_data["values"][i])

        result = {}
        for interaction_type, data in grouped.items():
            user_ids = torch.stack(data["users"])
            item_ids = torch.stack(data["items"])
            if len(data["values"]) == 1:
                values = data["values"][0]
                if values.dim() == 0:
                    values = values.unsqueeze(0)  # scalar -> [1]
                else:
                    values = values.unsqueeze(0)  # [output_dim] -> [1, output_dim]
            else:
                values = torch.stack(data["values"])
            result[interaction_type] = (user_ids, item_ids, values)

        return result

    @staticmethod
    def _align_shapes(
        predictions: torch.Tensor, targets: torch.Tensor, interaction_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align prediction/target dims, raising on genuine shape mismatches."""
        if predictions.dim() == 2 and targets.dim() == 1:
            if predictions.shape[1] == 1:
                targets = targets.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Shape mismatch for {interaction_type}: predictions {predictions.shape} "
                    f"vs targets {targets.shape}. Multi-dimensional predictions require "
                    f"multi-dimensional targets."
                )
        elif predictions.dim() == 1 and targets.dim() == 2:
            if targets.shape[1] == 1:
                targets = targets.squeeze(-1)
            else:
                raise ValueError(
                    f"Shape mismatch for {interaction_type}: predictions {predictions.shape} "
                    f"vs targets {targets.shape}. Single-dimensional predictions cannot match "
                    f"multi-dimensional targets."
                )
        return predictions, targets

    def _shared_parameters(self) -> List[torch.Tensor]:
        """Parameters shared across tasks, used for GradNorm balancing."""
        shared_parameters = []
        shared_parameters.extend(self.gnn_layers.parameters())
        shared_parameters.extend([self.user_embedding.weight, self.item_embedding.weight])
        shared_parameters.extend(self.relation_embedding.parameters())
        shared_parameters.extend([self.user_feature_proj.weight, self.item_feature_proj.weight])
        return shared_parameters

    def _sample_rating_negatives(
        self, user_ids: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Sample one negative item per entry in ``user_ids`` for BPR.

        A negative is an item the user has not positively rated (rating below
        ``rating_threshold``); when the per-user positive set is unavailable a
        uniformly random item is used.
        """
        pos_map = self.rating_pos_items or {}
        neg_items = []
        for u in user_ids.tolist():
            pos = pos_map.get(int(u), ())
            j = random.randrange(self.n_items)
            tries = 0
            while j in pos and tries < 10:
                j = random.randrange(self.n_items)
                tries += 1
            neg_items.append(j)
        return torch.tensor(neg_items, dtype=torch.long, device=device)

    def _compute_bpr_loss(
        self,
        pos_scores: torch.Tensor,
        anchor_user_ids: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        user_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """BPR pairwise ranking loss on RATING: -logsigmoid(s_pos - s_neg).

        Positives are the batch's above-threshold rated items; their predicted
        scores (``pos_scores``) are reused. Negatives are sampled per anchor and
        scored through the same RATING prediction path (graph head, or the
        causal SEM head for the causal models), so gradients flow consistently.
        """
        n_neg = max(1, int(self.bpr_num_neg))
        rep_users = anchor_user_ids.repeat_interleave(n_neg)
        neg_items = self._sample_rating_negatives(rep_users, pos_scores.device)
        neg_scores = self.predict_interaction(
            user_embeddings,
            item_embeddings,
            relation_embeddings,
            rep_users,
            neg_items,
            "RATING",
            user_features=user_features,
        ).view(-1)
        pos_rep = pos_scores.repeat_interleave(n_neg)
        return -F.logsigmoid(pos_rep - neg_scores).mean()

    def compute_loss(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        relation_embeddings: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
        interaction_weights: Dict[str, float] = None,
        return_individual_losses: bool = False,
        use_gradnorm: bool = False,
        gradnorm_handler=None,
        user_features=None,
    ) -> torch.Tensor:
        """
        Compute multi-task loss for heterogeneous interactions.

        Args:
            user_embeddings: User embeddings
            item_embeddings: Item embeddings
            relation_embeddings: Relation embeddings
            batch_data: Batch data containing user_ids, item_ids, ratings, interaction_types
            interaction_weights: Weights for different interaction types (used if not using GradNorm)
            return_individual_losses: Whether to return individual losses per interaction type
            use_gradnorm: Whether to use GradNorm for dynamic weight balancing
            gradnorm_handler: GradNorm instance for dynamic weight balancing
            user_features: User features, forwarded to predict_interaction for
                causal subclasses

        Returns:
            Total loss or tuple of (total_loss, individual_losses_dict) if return_individual_losses=True
        """
        # Reset the per-batch RATING component breakdown (set below in hybrid mode)
        self.last_rating_components = {}

        # Initialize individual_losses with all interaction types to ensure consistency
        individual_losses = {}
        for interaction_type in self.get_all_interaction_types():
            individual_losses[interaction_type] = torch.tensor(
                0.0, device=user_embeddings.device, requires_grad=True
            )

        # Compute individual losses for each interaction type in the batch
        for interaction_type, (user_ids, item_ids, targets) in self._group_batch_by_type(
            batch_data
        ).items():
            predictions = self.predict_interaction(
                user_embeddings,
                item_embeddings,
                relation_embeddings,
                user_ids,
                item_ids,
                interaction_type,
                user_features=user_features,
            )

            predictions, targets = self._align_shapes(predictions, targets, interaction_type)
            loss = F.mse_loss(predictions, targets)

            # Hybrid loss: add a BPR pairwise ranking term on RATING so training
            # also optimizes item ordering (recall/nDCG), while MSE retains the
            # rating magnitude the causal SEM head relies on. The MSE and (raw,
            # unweighted) BPR components are recorded detached for logging.
            if self.loss_mode == "hybrid" and interaction_type == "RATING":
                mse_component = loss
                bpr_component = predictions.new_tensor(0.0)
                pos_mask = targets.view(-1) >= self.rating_threshold
                if bool(pos_mask.any()):
                    pos_scores = predictions.view(-1)[pos_mask]
                    anchor_users = user_ids[pos_mask]
                    bpr_component = self._compute_bpr_loss(
                        pos_scores,
                        anchor_users,
                        user_embeddings,
                        item_embeddings,
                        relation_embeddings,
                        user_features,
                    )
                    loss = loss + self.bpr_lambda * bpr_component
                self.last_rating_components = {
                    "RATING_mse": float(mse_component.detach()),
                    "RATING_bpr": float(bpr_component.detach()),
                }

            if self.debug_mode:
                print(f"Loss computation for {interaction_type}:")
                print(
                    f"  Predictions: shape={predictions.shape}, range=[{predictions.min():.3f}, {predictions.max():.3f}]"
                )
                print(
                    f"  Targets: shape={targets.shape}, range=[{targets.min():.3f}, {targets.max():.3f}]"
                )
                print(f"  Loss: {loss.item():.6f}")

            individual_losses[interaction_type] = loss

        # Compute total loss using GradNorm or static weights
        if use_gradnorm and gradnorm_handler is not None:
            shared_parameters = self._shared_parameters()

            # Use GradNorm to compute weighted loss
            from utility.gradnorm import compute_weighted_loss

            total_loss, updated_weights = compute_weighted_loss(
                individual_losses,
                interaction_weights or {},
                use_gradnorm=True,
                gradnorm_handler=gradnorm_handler,
                shared_parameters=shared_parameters,
            )

            # Convert losses to float for return if needed
            if return_individual_losses:
                individual_losses_float = {}
                for k, v in individual_losses.items():
                    if hasattr(v, "item"):
                        individual_losses_float[k] = v.item()
                    else:
                        individual_losses_float[k] = v
                individual_losses = individual_losses_float
        else:
            # Use static weights
            individual_losses_float = {}
            total_loss = 0.0
            for interaction_type, loss in individual_losses.items():
                # Store individual loss value before applying weights
                if return_individual_losses:
                    individual_losses_float[interaction_type] = (
                        loss.item() if hasattr(loss, "item") else loss
                    )

                # Apply interaction weight
                if interaction_weights and interaction_type in interaction_weights:
                    loss = loss * interaction_weights[interaction_type]
                total_loss += loss

            # Update individual_losses with float values if needed
            if return_individual_losses:
                individual_losses = individual_losses_float
            else:
                individual_losses = {
                    k: v.item() if hasattr(v, "item") else v for k, v in individual_losses.items()
                }

        # Add regularization
        reg_loss = self._compute_regularization_loss()
        total_loss += 1e-5 * reg_loss

        if return_individual_losses:
            return total_loss, individual_losses
        return total_loss

    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss."""
        reg_loss = 0.0

        # Regularize embeddings
        reg_loss += torch.norm(self.user_embedding.weight, p=2)
        reg_loss += torch.norm(self.item_embedding.weight, p=2)
        reg_loss += torch.norm(self.relation_embedding.relation_embeddings.weight, p=2)

        # Regularize GNN layer parameters
        for layer in self.gnn_layers:
            for param in layer.parameters():
                reg_loss += torch.norm(param, p=2)

        return reg_loss
