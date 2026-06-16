"""
GHCF_Causal: GHCF model where non-BFI interactions are predicted causally.

The GNN is built on BFI_PERCEPTION interactions only. BFI predictions from
the graph are turned into BFI-TIPI differences and propagated through the
differentiable causal SEM head (CausalMixin) to predict all other
interaction types. The entire path is differentiable, so rating/quality/
feeling losses train the GNN through the causal structure.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.CausalMixin import CausalMixin
    from models.GHCF import GHCF
except ImportError:
    from CausalMixin import CausalMixin
    from GHCF import GHCF


class GHCF_Causal(GHCF, CausalMixin):
    """
    Causal GHCF model that predicts non-BFI interactions through the causal DAG.

    This model extends the base GHCF by:
    1. Building the GNN on BFI_PERCEPTION interactions only
    2. Computing BFI-TIPI differences from graph BFI predictions
    3. Predicting all other interaction values via the SEM head
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
        use_causal: bool = True,
        causal_model_path: str = None,
        causal_config_path: str = None,
    ):
        """
        Initialize GHCF_Causal model.

        Args:
            n_users: Number of users
            n_items: Number of items
            n_user_features: Number of user features
            n_item_features: Number of item features
            interaction_types: List of interaction type names (must include BFI_PERCEPTION)
            embedding_dim: Embedding dimension
            layer_sizes: Sizes of GNN layers
            dropout_rates: Dropout rates for each layer
            gnn_type: Type of GNN layer ('gcn', 'sage', 'gat')
            n_heads: Number of attention heads (only used for 'gat')
            device: Device to run the model on
            use_causal: Whether to use causal prediction
            causal_model_path: Path to trained causal model
            causal_config_path: Path to causal model configuration
        """
        # Store all interaction types for loss computation
        self.all_interaction_types = interaction_types.copy()

        if "BFI_PERCEPTION" not in interaction_types:
            raise ValueError(
                "BFI_PERCEPTION must be included in interaction_types for GHCF_Causal"
            )

        # The GNN only models BFI_PERCEPTION; everything else is causal
        super(GHCF_Causal, self).__init__(
            n_users=n_users,
            n_items=n_items,
            n_user_features=n_user_features,
            n_item_features=n_item_features,
            interaction_types=["BFI_PERCEPTION"],
            embedding_dim=embedding_dim,
            layer_sizes=layer_sizes,
            dropout_rates=dropout_rates,
            gnn_type=gnn_type,
            n_heads=n_heads,
            device=device,
        )

        # Initialize causal components from mixin
        self.init_causal_components(use_causal, causal_model_path, causal_config_path)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        adj_matrices: Dict[str, torch.sparse.FloatTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass using only BFI adjacency matrices for GNN computation.

        Args:
            user_features: User features [n_users, n_user_features]
            item_features: Item features [n_items, n_item_features]
            adj_matrices: Dictionary of adjacency matrices for each interaction type

        Returns:
            Tuple of (user_embeddings, item_embeddings, relation_embeddings)
        """
        bfi_adj_matrices = {k: v for k, v in adj_matrices.items() if "BFI" in k.upper()}

        if not bfi_adj_matrices:
            raise ValueError("No BFI adjacency matrices found in adj_matrices")

        return super().forward(user_features, item_features, bfi_adj_matrices)

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
        Predict interaction scores: BFI via the graph, everything else causally.

        Args:
            user_embeddings: User embeddings [n_users, embedding_dim]
            item_embeddings: Item embeddings [n_items, embedding_dim]
            relation_embeddings: Relation embeddings [n_relations, embedding_dim]
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            interaction_type: Type of interaction to predict
            component_idx: Specific component index (BFI predictions only)
            user_features: User features for causal prediction [n_users, n_features]

        Returns:
            Predicted interaction scores [batch_size] or [batch_size, output_dim]
        """
        if interaction_type == "BFI_PERCEPTION":
            return super().predict_interaction(
                user_embeddings,
                item_embeddings,
                relation_embeddings,
                user_ids,
                item_ids,
                interaction_type,
                component_idx=component_idx,
            )

        # For all other interactions, use causal prediction
        self.validate_causal_requirements(interaction_type, user_features)

        bfi_prediction = super().predict_interaction(
            user_embeddings,
            item_embeddings,
            relation_embeddings,
            user_ids,
            item_ids,
            "BFI_PERCEPTION",
        )

        bfi_diff = self._compute_bfi_tipi_difference(bfi_prediction, user_features, user_ids)
        causal_predictions = self._predict_with_causal_model(
            bfi_diff, self.all_interaction_types, debug=self.debug_mode
        )

        if interaction_type not in causal_predictions:
            raise ValueError(
                f"Causal prediction not available for interaction type: {interaction_type}"
            )

        return causal_predictions[interaction_type]

    def get_all_interaction_types(self) -> List[str]:
        """
        Get all interaction types that this model can predict.

        Returns:
            List of all interaction types including those predicted causally
        """
        return self.all_interaction_types.copy()
