"""
GHCF_Refine: GHCF model with optional causal refinement of predictions.

The GNN is built on all interaction types. When causal mode is enabled and
user features are available, non-BFI predictions are replaced by the
differentiable causal SEM head (CausalMixin) driven by BFI-TIPI differences;
otherwise the model falls back to the graph-based predictions.
"""

import sys
from pathlib import Path
from typing import List

import torch

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.CausalMixin import CausalMixin
    from models.GHCF import GHCF
except ImportError:
    from CausalMixin import CausalMixin
    from GHCF import GHCF


class GHCF_Refine(GHCF, CausalMixin):
    """
    Enhanced GHCF model with optional causal prediction capabilities.

    This model extends the base GHCF by:
    1. Optionally predicting non-BFI interactions via the causal SEM head
    2. Computing BFI-TIPI differences when causal mode is enabled
    3. Falling back to graph-based predictions when causal is unavailable
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
        use_causal: bool = False,
        causal_model_path: str = None,
        causal_config_path: str = None,
    ):
        """
        Initialize GHCF_Refine model.

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
            use_causal: Whether to use causal prediction
            causal_model_path: Path to trained causal model
            causal_config_path: Path to causal model configuration
        """
        super(GHCF_Refine, self).__init__(
            n_users=n_users,
            n_items=n_items,
            n_user_features=n_user_features,
            n_item_features=n_item_features,
            interaction_types=interaction_types,
            embedding_dim=embedding_dim,
            layer_sizes=layer_sizes,
            dropout_rates=dropout_rates,
            gnn_type=gnn_type,
            n_heads=n_heads,
            device=device,
        )

        # Initialize causal components from mixin
        self.init_causal_components(use_causal, causal_model_path, causal_config_path)

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
        Predict interaction scores with optional causal refinement.

        Args:
            user_embeddings: User embeddings [n_users, embedding_dim]
            item_embeddings: Item embeddings [n_items, embedding_dim]
            relation_embeddings: Relation embeddings [n_relations, embedding_dim]
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            interaction_type: Type of interaction to predict
            component_idx: Specific component index (graph predictions only)
            user_features: User features for causal prediction [n_users, n_features]

        Returns:
            Predicted interaction scores [batch_size] or [batch_size, output_dim]
        """
        # BFI is always predicted by the graph model; other types fall back to
        # the graph prediction when causal mode is unavailable
        use_causal_path = (
            interaction_type != "BFI_PERCEPTION"
            and self.use_causal
            and user_features is not None
            and self.causal_head is not None
        )

        if not use_causal_path:
            return super().predict_interaction(
                user_embeddings,
                item_embeddings,
                relation_embeddings,
                user_ids,
                item_ids,
                interaction_type,
                component_idx=component_idx,
            )

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
            bfi_diff, self.interaction_types, debug=self.debug_mode
        )

        if interaction_type not in causal_predictions:
            raise ValueError(
                f"Causal prediction not available for interaction type: {interaction_type}"
            )

        return causal_predictions[interaction_type]
