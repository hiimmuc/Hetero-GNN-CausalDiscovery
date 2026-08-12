"""
CausalMixin: Mixin providing causal prediction capabilities for GHCF models.

The mixin owns a differentiable SEMHead (see CausalHead.py) built from a
fitted LiNGAM model. The graph model predicts BFI perception; the mixin
computes BFI-TIPI differences and propagates them through the causal DAG to
predict rating, voice-quality, and subjective-feeling interactions. The whole
path stays inside autograd so those losses train the GNN end-to-end.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.CausalHead import SEMHead, load_sem_head
except ImportError:
    from CausalHead import SEMHead, load_sem_head


def _resolve_model_data_path(causal_model_path: str) -> Path:
    """Resolve any saved-model filename variant to its *_model_data.npy file."""
    path = str(causal_model_path)
    if path.endswith("_model_data.npy"):
        return Path(path)
    for suffix in ("_robust_adjacency_matrix.npy", "_adjacency_matrix.npy"):
        if path.endswith(suffix):
            prefix = "_robust" if suffix.startswith("_robust") else ""
            return Path(path[: -len(suffix)] + f"{prefix}_model_data.npy")
    raise ValueError(
        f"Cannot resolve causal model data from '{causal_model_path}'. "
        f"Expected a *_model_data.npy or *_adjacency_matrix.npy file."
    )


def _resolve_norm_stats_path(model_data_path: Path) -> Path:
    """Resolve the normalization stats CSV that accompanies a model data file."""
    name = model_data_path.name
    stats_name = name.replace("_robust_model_data.npy", "_normalization_stats.csv").replace(
        "_model_data.npy", "_normalization_stats.csv"
    )
    return model_data_path.parent / stats_name


class CausalMixin:
    """
    Mixin class providing causal prediction capabilities for GHCF models.

    Provides:
    1. Loading of the fitted LiNGAM model into a differentiable SEMHead
    2. BFI-TIPI difference computation (input normalization lives in SEMHead)
    3. Causal prediction of non-BFI interaction types from BFI differences
    """

    def init_causal_components(
        self,
        use_causal: bool = False,
        causal_model_path: str | None = None,
        causal_config_path: str | None = None,
        trainable_sem: bool = True,
    ) -> None:
        """
        Initialize causal prediction components.

        Args:
            use_causal: Whether to enable causal prediction
            causal_model_path: Path to a saved LiNGAM model
                (*_model_data.npy or *_adjacency_matrix.npy)
            causal_config_path: Path to causal model configuration (kept for
                reproducibility logging; not required to build the head)
            trainable_sem: Whether SEM coefficients are fine-tuned during
                training (DAG mask is fixed either way)
        """
        self.use_causal = use_causal
        self.causal_head: SEMHead | None = None
        self.causal_config_path = causal_config_path

        if not use_causal:
            return
        if not causal_model_path:
            raise ValueError("use_causal=True requires causal_model_path")

        model_data_path = _resolve_model_data_path(causal_model_path)
        norm_stats_path = _resolve_norm_stats_path(model_data_path)

        self.causal_head = load_sem_head(
            str(model_data_path),
            norm_stats_path=str(norm_stats_path),
            trainable=trainable_sem,
        )
        print(
            f"Loaded causal SEM head from {model_data_path} "
            f"({int(self.causal_head.mask.sum().item())} causal edges, "
            f"trainable={trainable_sem})"
        )

        # Guard against a robust-adjacency matrix that pruned the causal path.
        # forward() would silently emit constant zeros for any disconnected
        # output; surface that instead of training against dead targets.
        disconnected = self.causal_head.disconnected_outputs()
        if disconnected:
            msg = (
                f"causal SEM head has output(s) with no path from the BFI-TIPI "
                f"inputs: {disconnected}. forward() predicts these as constant "
                f"zero. This usually means the robust-adjacency significance "
                f"threshold pruned too many edges; re-run run_causal_discovery.py "
                f"with a lower threshold or point --causal_path at the non-robust "
                f"model (*_model_data.npy without '_robust')."
            )
            if len(disconnected) == self.causal_head.n_output_components():
                raise ValueError("Entire " + msg)
            print("[causal][WARNING] Partial " + msg)

    def causal_trainable_parameters(self) -> list[torch.Tensor]:
        """Trainable parameters of the causal SEM head (the masked SEM weights).

        These form the shared causal trunk for every non-BFI task, so GradNorm
        should balance task gradients through them. Empty when the head is
        frozen (its ``W`` is then a buffer, not a parameter) or absent.
        """
        if getattr(self, "causal_head", None) is None:
            return []
        return [p for p in self.causal_head.parameters() if p.requires_grad]

    def _compute_bfi_tipi_difference(
        self, bfi_predictions: torch.Tensor, user_features: torch.Tensor, user_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute difference between BFI predictions and TIPI features.

        The z-score normalization of the difference is owned by SEMHead.

        Args:
            bfi_predictions: BFI predictions from graph model [batch_size, 5]
            user_features: User features containing TIPI [n_users, n_features]
            user_ids: User IDs for this batch [batch_size]

        Returns:
            Raw BFI-TIPI difference [batch_size, 5]
        """
        # Last 5 user features are the TIPI scores (see dataloader feature order)
        batch_user_features = user_features[user_ids]
        tipi_features = batch_user_features[:, -5:]
        bfi_predictions = self._ensure_bfi_shape(bfi_predictions)

        return bfi_predictions - tipi_features

    def _ensure_bfi_shape(self, bfi_prediction: torch.Tensor) -> torch.Tensor:
        """
        Validate that BFI predictions have shape [batch_size, 5].

        Args:
            bfi_prediction: BFI prediction tensor

        Returns:
            BFI prediction [batch_size, 5]
        """
        if torch.isnan(bfi_prediction).any():
            raise ValueError("NaN detected in BFI prediction")

        if bfi_prediction.dim() == 1:
            bfi_prediction = bfi_prediction.unsqueeze(0)

        if bfi_prediction.dim() != 2 or bfi_prediction.shape[-1] != 5:
            raise ValueError(
                f"BFI prediction must have shape [batch_size, 5], "
                f"got {tuple(bfi_prediction.shape)}"
            )

        return bfi_prediction

    def _predict_with_causal_model(
        self, bfi_diff: torch.Tensor, interaction_types: list[str], debug: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Predict interaction values from BFI-TIPI differences via the SEM head.

        Args:
            bfi_diff: Raw BFI-TIPI difference [batch_size, 5]
            interaction_types: Interaction types to predict
            debug: Whether to print debug information

        Returns:
            Dictionary of predicted interaction values (normalized space)
        """
        if self.causal_head is None:
            raise ValueError("Causal SEM head not loaded")

        if debug:
            print(
                f"BFI diff shape: {bfi_diff.shape}, "
                f"range: [{bfi_diff.min():.3f}, {bfi_diff.max():.3f}]"
            )

        all_predictions = self.causal_head(bfi_diff)
        predictions = {
            interaction_type: all_predictions[interaction_type]
            for interaction_type in interaction_types
            if interaction_type in all_predictions
        }

        if debug:
            for int_type, pred in predictions.items():
                print(
                    f"Causal prediction for {int_type}: shape {tuple(pred.shape)}, "
                    f"range [{pred.min():.3f}, {pred.max():.3f}]"
                )

        return predictions

    def validate_causal_requirements(
        self, interaction_type: str, user_features: torch.Tensor | None
    ) -> None:
        """
        Validate requirements for causal prediction.

        Args:
            interaction_type: The interaction type to predict
            user_features: User features tensor

        Raises:
            ValueError: If requirements are not met
        """
        if not self.use_causal:
            raise ValueError(
                f"Causal prediction is disabled but required for interaction type: "
                f"{interaction_type}"
            )

        if user_features is None:
            raise ValueError(
                f"User features are required for causal prediction of interaction type: "
                f"{interaction_type}. Make sure to pass user_features to compute_loss "
                f"or predict_interaction methods."
            )
