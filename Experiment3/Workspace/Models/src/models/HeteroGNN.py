"""
Heterogeneous Graph Neural Network components for multi-relational recommendation.
"""

import torch
import torch.nn.functional as F
from torch import nn


class HeteroGCNLayer(nn.Module):
    """
    Heterogeneous Graph Convolutional Network Layer for multi-relational data.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int, dropout: float = 0.1):
        """
        Initialize HeteroGCN layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            n_relations: Number of relation types
            dropout: Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations

        # Separate transformations for each relation type
        self.relation_transforms = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=True) for _ in range(n_relations)]
        )

        # Self-loop transformation
        self.self_transform = nn.Linear(in_dim, out_dim, bias=True)

        # Relation attention weights
        self.relation_attention = nn.Parameter(torch.ones(n_relations))

        # GHCF-style per-layer relation transform (W_rel): refines the relation
        # embeddings as they propagate through layers (shared across relations).
        self.relation_layer_transform = nn.Linear(in_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)  # GHCF uses leaky_relu in propagation

    def forward(
        self,
        x: torch.Tensor,
        adj_matrices: list[torch.sparse.FloatTensor],
        relation_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through HeteroGCN layer.

        Args:
            x: Node features [n_nodes, in_dim]
            adj_matrices: List of adjacency matrices for each relation
            relation_embeddings: Per-relation embeddings [n_relations, in_dim].
                When provided, neighbor messages for relation i are modulated
                element-wise by relation_embeddings[i] (GHCF: (A@E) ⊙ r) before
                the relation transform.

        Returns:
            Tuple of (updated node features [n_nodes, out_dim],
            relation embeddings propagated to the next layer [n_relations,
            out_dim] or None if no relation embeddings were given)
        """
        # Self-loop features
        self_features = self.self_transform(x)

        # Aggregate features from different relations
        relation_features = []

        for i, adj_matrix in enumerate(adj_matrices):
            if adj_matrix is not None:
                # Message passing for this relation
                neighbor_features = torch.sparse.mm(adj_matrix, x)

                # GHCF: modulate neighbor messages by the relation embedding
                if relation_embeddings is not None:
                    neighbor_features = neighbor_features * relation_embeddings[i].unsqueeze(0)

                transformed_features = self.relation_transforms[i](neighbor_features)

                # Apply relation attention
                attention_weight = torch.softmax(self.relation_attention, dim=0)[i]
                weighted_features = attention_weight * transformed_features

                relation_features.append(weighted_features)

        # Combine features from all relations
        if relation_features:
            aggregated_features = torch.stack(relation_features, dim=0).mean(dim=0)
        else:
            aggregated_features = torch.zeros_like(self_features)

        # Combine with self features
        output = self_features + aggregated_features
        output = self.dropout(self.activation(output))

        # Propagate relation embeddings to the next layer (GHCF: r <- r @ W_rel)
        new_relation_embeddings = (
            self.relation_layer_transform(relation_embeddings)
            if relation_embeddings is not None
            else None
        )

        return output, new_relation_embeddings


class HeteroGraphSAGELayer(nn.Module):
    """
    Heterogeneous GraphSAGE layer with mean aggregation.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int, dropout: float = 0.1):
        """
        Initialize HeteroGraphSAGE layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            n_relations: Number of relation types
            dropout: Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations

        # Neighbor aggregation for each relation
        self.neighbor_transforms = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=True) for _ in range(n_relations)]
        )

        # Self transformation
        self.self_transform = nn.Linear(in_dim, out_dim, bias=True)

        # Combine transformation
        self.combine_transform = nn.Linear(out_dim * 2, out_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        adj_matrices: list[torch.sparse.FloatTensor],
        relation_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through HeteroGraphSAGE layer.

        Args:
            x: Node features [n_nodes, in_dim]
            adj_matrices: List of adjacency matrices for each relation
            relation_embeddings: Per-relation embeddings, passed through
                unchanged (this non-GHCF variant does not modulate by them).

        Returns:
            Tuple of (updated node features [n_nodes, out_dim],
            relation_embeddings unchanged)
        """
        # Self features
        self_features = self.self_transform(x)

        # Aggregate neighbor features from all relations
        neighbor_features_list = []

        for i, adj_matrix in enumerate(adj_matrices):
            if adj_matrix is not None:
                # Mean aggregation of neighbors
                neighbor_features = torch.sparse.mm(adj_matrix, x)
                transformed_neighbors = self.neighbor_transforms[i](neighbor_features)
                neighbor_features_list.append(transformed_neighbors)

        # Combine neighbor features from all relations
        if neighbor_features_list:
            combined_neighbors = torch.stack(neighbor_features_list, dim=0).mean(dim=0)
        else:
            combined_neighbors = torch.zeros_like(self_features)

        # Concatenate self and neighbor features
        combined_features = torch.cat([self_features, combined_neighbors], dim=1)

        # Final transformation
        output = self.combine_transform(combined_features)
        output = self.dropout(self.activation(output))

        return output, relation_embeddings


class HeteroGATLayer(nn.Module):
    """
    Heterogeneous Graph Attention Network layer.
    """

    def __init__(
        self, in_dim: int, out_dim: int, n_relations: int, n_heads: int = 4, dropout: float = 0.1
    ):
        """
        Initialize HeteroGAT layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            n_relations: Number of relation types
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        # Attention mechanisms: independent projection + attention vector per
        # (relation, head) so the heads learn distinct attention patterns.
        self.attention_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                nn.Linear(in_dim, self.head_dim, bias=False),
                                nn.Linear(2 * self.head_dim, 1, bias=False),
                            ]
                        )
                        for _ in range(n_heads)
                    ]
                )
                for _ in range(n_relations)
            ]
        )

        # Value transformations
        self.value_transforms = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=True) for _ in range(n_relations)]
        )

        # Self transformation
        self.self_transform = nn.Linear(in_dim, out_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        adj_matrices: list[torch.sparse.FloatTensor],
        relation_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through HeteroGAT layer.

        Args:
            x: Node features [n_nodes, in_dim]
            adj_matrices: List of adjacency matrices for each relation
            relation_embeddings: Per-relation embeddings, passed through
                unchanged (this non-GHCF variant does not modulate by them).

        Returns:
            Tuple of (updated node features [n_nodes, out_dim],
            relation_embeddings unchanged)
        """
        n_nodes = x.size(0)
        self_features = self.self_transform(x)

        relation_outputs = []

        for rel_idx, adj_matrix in enumerate(adj_matrices):
            if adj_matrix is None:
                continue

            # Multi-head attention for this relation
            head_outputs = []

            for head in range(self.n_heads):
                # Each head has its own projection and attention vector
                head_transform = self.attention_layers[rel_idx][head][0]
                attention_weight = self.attention_layers[rel_idx][head][1]

                x_head = head_transform(x)  # [n_nodes, head_dim]

                # Calculate attention scores
                edge_indices = adj_matrix.coalesce().indices()
                src_nodes = edge_indices[0]
                tgt_nodes = edge_indices[1]

                src_features = x_head[src_nodes]  # [n_edges, head_dim]
                tgt_features = x_head[tgt_nodes]  # [n_edges, head_dim]

                # Attention mechanism
                edge_features = torch.cat([src_features, tgt_features], dim=1)
                attention_scores = attention_weight(edge_features).squeeze(-1)
                attention_scores = F.leaky_relu(attention_scores, 0.2)

                # Softmax attention per target node
                attention_probs = torch.zeros(adj_matrix.size(0), device=x.device)
                attention_probs.index_add_(0, tgt_nodes, attention_scores.exp())
                attention_probs = attention_probs[tgt_nodes]
                attention_weights = attention_scores.exp() / (attention_probs + 1e-8)

                # Apply attention to source features
                weighted_features = attention_weights.unsqueeze(1) * src_features

                # Aggregate for each target node
                head_output = torch.zeros(n_nodes, self.head_dim, device=x.device)
                head_output.index_add_(0, tgt_nodes, weighted_features)

                head_outputs.append(head_output)

            # Concatenate heads
            if head_outputs:
                relation_output = torch.cat(head_outputs, dim=1)
                relation_outputs.append(relation_output)

        # Combine outputs from all relations
        if relation_outputs:
            aggregated_output = torch.stack(relation_outputs, dim=0).mean(dim=0)
        else:
            aggregated_output = torch.zeros_like(self_features)

        # Combine with self features
        output = self_features + aggregated_output
        output = self.dropout(self.activation(output))

        return output, relation_embeddings


class MultiRelationEmbedding(nn.Module):
    """
    Multi-relation embedding layer for heterogeneous interactions.
    """

    def __init__(self, n_relations: int, embedding_dim: int):
        """
        Initialize multi-relation embedding.

        Args:
            n_relations: Number of relation types
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim

        # Relation embeddings
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """
        Get relation embeddings.

        Args:
            relation_ids: Relation IDs [batch_size] or []

        Returns:
            Relation embeddings [batch_size, embedding_dim] or [n_relations, embedding_dim]
        """
        if relation_ids.numel() == 0:
            # Return all relation embeddings
            return self.relation_embeddings.weight
        else:
            return self.relation_embeddings(relation_ids)
