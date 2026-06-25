"""
GHCF_ACE: Adaptive Causal Ensemble GHCF.

The GNN is built on *all* interaction types (15 relations), so every
interaction (including RATING / VOICE_QUALITY / SUBJECTIVE_FEELING) gets a
graph-based prediction. In parallel, the BFI-TIPI difference is propagated
through the differentiable causal SEM head (CausalMixin, exactly as in
GHCF-CE) to produce a causal prediction for those same non-BFI interactions.

The final non-BFI prediction is a per-interaction-type weighted combination of
the two sources:

    pred = w_mlp * pred_gnn + w_causal * pred_causal

The weights are either:
  * "learnable": a per-type gate, w_causal = sigmoid(logit), w_mlp = 1 - w_causal
    (convex, attention-like; the learned value is interpretable), or
  * "fixed":     per-type hyperparameters (w_mlp, w_causal) for manual tuning.

BFI_PERCEPTION itself is always predicted by the graph (it is the input to the
causal head, not an output of it). Both prediction sources live in the same
normalized target space (the dataloader z-scores RATING/VQ/SF with the same
stats the SEM head uses), so the weighted average is well defined.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.CausalMixin import CausalMixin
    from models.GHCF import GHCF
except ImportError:
    from CausalMixin import CausalMixin
    from GHCF import GHCF


class GHCF_ACE(GHCF, CausalMixin):
    """
    Adaptive Causal Ensemble GHCF.

    Extends the base GHCF by:
    1. Building the GNN on all interaction types (graph predicts everything).
    2. Also predicting non-BFI interactions via the causal SEM head from the
       BFI-TIPI difference (as in GHCF-CE).
    3. Combining the two via per-type weights (learnable gate or fixed).
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
        ace_weight_mode: str = "learnable",
        ace_fixed_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize GHCF_ACE model.

        Args:
            ace_weight_mode: "learnable" (per-type sigmoid gate) or "fixed"
                (per-type hyperparameters).
            ace_fixed_weights: {interaction_type: {"w_mlp": .., "w_causal": ..}}
                used only when ace_weight_mode == "fixed". Missing types default
                to 0.5 / 0.5.
            (other args as in GHCF.)
        """
        self.all_interaction_types = interaction_types.copy()

        if "BFI_PERCEPTION" not in interaction_types:
            raise ValueError(
                "BFI_PERCEPTION must be included in interaction_types for GHCF_ACE"
            )

        # The GNN models all interaction types (unlike GHCF_CE).
        super(GHCF_ACE, self).__init__(
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

        # Causal SEM head (same path as GHCF-CE)
        self.init_causal_components(use_causal, causal_model_path, causal_config_path)

        # Per-type GNN/causal mixing weights
        self._init_ace_weights(ace_weight_mode, ace_fixed_weights)

    def _init_ace_weights(
        self, mode: str, fixed_weights: Optional[Dict[str, Dict[str, float]]]
    ) -> None:
        """Initialize the per-type GNN/causal mixing weights."""
        if mode not in ("learnable", "fixed"):
            raise ValueError(f"ace_weight_mode must be 'learnable' or 'fixed', got '{mode}'")

        self.ace_weight_mode = mode
        # Types blended by the causal head (everything the SEM head predicts)
        self.ace_types = [t for t in self.all_interaction_types if t != "BFI_PERCEPTION"]

        if mode == "learnable":
            # logit per type; sigmoid(0) = 0.5 -> equal blend at initialization
            self.ace_gate_logits = nn.ParameterDict(
                {t: nn.Parameter(torch.zeros(1)) for t in self.ace_types}
            )
        else:  # fixed
            fixed_weights = fixed_weights or {}
            self.ace_fixed_weights: Dict[str, Tuple[float, float]] = {}
            for t in self.ace_types:
                w = fixed_weights.get(t, {})
                self.ace_fixed_weights[t] = (
                    float(w.get("w_mlp", 0.5)),
                    float(w.get("w_causal", 0.5)),
                )

    def _ace_weights(self, interaction_type: str):
        """Return (w_mlp, w_causal) for an interaction type.

        Learnable mode returns tensors (a convex gate); fixed mode returns
        Python floats. Both broadcast against prediction tensors.
        """
        if self.ace_weight_mode == "learnable":
            w_causal = torch.sigmoid(self.ace_gate_logits[interaction_type])
            return (1.0 - w_causal), w_causal
        return self.ace_fixed_weights[interaction_type]

    def get_ace_weights(self) -> Dict[str, Tuple[float, float]]:
        """Current (w_mlp, w_causal) per type as floats, for logging/inspection."""
        out: Dict[str, Tuple[float, float]] = {}
        for t in self.ace_types:
            w_mlp, w_causal = self._ace_weights(t)
            out[t] = (
                float(w_mlp.item() if torch.is_tensor(w_mlp) else w_mlp),
                float(w_causal.item() if torch.is_tensor(w_causal) else w_causal),
            )
        return out

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
        Predict interaction scores.

        BFI is the graph prediction. Non-BFI interactions blend the graph head
        with the causal SEM head; when causal mode is unavailable they fall back
        to the graph prediction (i.e. the model degrades to the base GHCF).
        """
        # Graph (GNN) prediction for this interaction type
        gnn_prediction = super().predict_interaction(
            user_embeddings,
            item_embeddings,
            relation_embeddings,
            user_ids,
            item_ids,
            interaction_type,
            component_idx=component_idx,
        )

        # BFI is graph-only; it is the input to the causal head, not an output
        if interaction_type == "BFI_PERCEPTION":
            return gnn_prediction

        # Fall back to the graph prediction when causal prediction is unavailable
        causal_available = (
            self.use_causal and user_features is not None and self.causal_head is not None
        )
        if not causal_available:
            return gnn_prediction

        # Causal SEM-head prediction from the BFI-TIPI difference
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
            return gnn_prediction

        causal_prediction = causal_predictions[interaction_type]
        if component_idx is not None:
            causal_prediction = causal_prediction[:, component_idx : component_idx + 1]

        # Weighted ensemble of the two sources
        w_mlp, w_causal = self._ace_weights(interaction_type)
        return w_mlp * gnn_prediction + w_causal * causal_prediction

    def get_all_interaction_types(self) -> List[str]:
        """All interaction types this model can predict."""
        return self.all_interaction_types.copy()
