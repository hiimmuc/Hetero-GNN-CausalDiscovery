"""
Differentiable structural-equation-model (SEM) prediction head.

Wraps the DAG discovered by LiNGAM (Stage 1) as a torch module so that
predictions for mediator and outcome variables are produced by propagating
the BFI-TIPI difference inputs through the causal graph *inside* autograd.
Weights are initialized from the LiNGAM coefficients and masked by the
discovered DAG structure (NOTEARS-MLP-style masking on a linear SEM), so
gradients from rating/quality/feeling losses flow back into the GNN while
the causal structure is preserved.

Predictions are in the same normalized (z-scored) space the LiNGAM model
was fitted on, matching the normalized interaction targets produced by the
dataloader.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# Exogenous input features, in the column order of the bfi_diff tensor (O, C, E, A, N)
INPUT_FEATURES = [
    "Difference BFI-TIPI O",
    "Difference BFI-TIPI C",
    "Difference BFI-TIPI E",
    "Difference BFI-TIPI A",
    "Difference BFI-TIPI N",
]


class SEMHead(nn.Module):
    """
    Linear SEM head constrained by a discovered causal DAG.

    The structural equations are x_v = sum_j W[v, j] * x_j over the parents
    of v, evaluated in causal order so mediated paths (e.g. BFI diff ->
    MOS-X2/SQ -> Rating) are propagated correctly.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        features: Sequence[str],
        causal_order: Sequence[int],
        interaction_columns: dict[str, list[str]],
        trainable: bool = True,
        norm_stats: pd.DataFrame | None = None,
    ):
        """
        Initialize SEM head from a fitted LiNGAM model.

        Args:
            adjacency_matrix: LiNGAM adjacency matrix [n_features, n_features],
                A[i, j] = causal effect of feature j on feature i
            features: Feature names matching the matrix rows/columns
            causal_order: Topological order of feature indices from LiNGAM
            interaction_columns: Mapping interaction_type -> list of feature
                (column) names that interaction predicts
            trainable: Whether the SEM coefficients are fine-tuned during
                training (masked to the discovered DAG either way)
            norm_stats: DataFrame with columns [feature, mean, std] used to
                z-score the BFI-diff inputs before propagation
        """
        super().__init__()

        adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.float32)
        n_features = adjacency_matrix.shape[0]
        if adjacency_matrix.shape != (n_features, n_features):
            raise ValueError(f"Adjacency matrix must be square, got {adjacency_matrix.shape}")
        if len(features) != n_features:
            raise ValueError(
                f"Got {len(features)} feature names for adjacency matrix of size {n_features}"
            )

        self.features = list(features)
        self.causal_order = [int(v) for v in causal_order]
        self.n_features = n_features

        # Input (exogenous) feature indices in bfi_diff column order
        self.input_indices = []
        for name in INPUT_FEATURES:
            if name not in self.features:
                raise ValueError(f"Input feature '{name}' not found in causal features")
            self.input_indices.append(self.features.index(name))

        # Map each interaction type to the feature indices it predicts
        self.interaction_indices: dict[str, list[int]] = {}
        for interaction_type, columns in interaction_columns.items():
            if interaction_type == "BFI_PERCEPTION":
                continue  # BFI perception is predicted by the graph model
            missing = [c for c in columns if c not in self.features]
            if missing:
                raise ValueError(
                    f"Columns {missing} for {interaction_type} not found in causal features "
                    f"{self.features}"
                )
            self.interaction_indices[interaction_type] = [self.features.index(c) for c in columns]

        # DAG-masked SEM coefficients, initialized from LiNGAM
        weight = torch.from_numpy(adjacency_matrix)
        if trainable:
            self.W = nn.Parameter(weight)
        else:
            self.register_buffer("W", weight)
        self.register_buffer("mask", (weight != 0).float())

        # Precompute parent lists per variable (from the fixed mask)
        self._parents = {
            v: torch.nonzero(self.mask[v], as_tuple=True)[0].tolist() for v in range(n_features)
        }

        # Reachability: a feature is "driven" if it is an exogenous input or has
        # at least one driven parent. forward() emits constant zeros for any
        # non-input feature with no (driven) parents, so an undriven output means
        # its causal path was pruned away (e.g. by the robust-adjacency
        # significance thresholding). Parents precede children in causal_order,
        # so a single forward pass resolves reachability.
        driven = [False] * n_features
        for idx in self.input_indices:
            driven[idx] = True
        for v in self.causal_order:
            if not driven[v] and any(driven[j] for j in self._parents[v]):
                driven[v] = True
        self._driven = driven

        # Normalization stats for the input features (z-score before propagation)
        input_mean = torch.zeros(len(self.input_indices))
        input_std = torch.ones(len(self.input_indices))
        if norm_stats is not None:
            stats = norm_stats.set_index("feature")
            for i, name in enumerate(INPUT_FEATURES):
                if name not in stats.index:
                    raise ValueError(f"Normalization stats missing for feature '{name}'")
                input_mean[i] = float(stats.loc[name, "mean"])
                input_std[i] = float(stats.loc[name, "std"])
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)

    def n_output_components(self) -> int:
        """Total number of predicted output components across interaction types."""
        return sum(len(indices) for indices in self.interaction_indices.values())

    def disconnected_outputs(self) -> list[str]:
        """Names of predicted outputs with no causal path from the BFI-TIPI inputs.

        forward() emits constant zeros for these, so a nonempty list means part
        of the causal graph was pruned (commonly by robust-adjacency
        thresholding). Returned as ``interaction_type:feature`` labels.
        """
        names: list[str] = []
        for interaction_type, indices in self.interaction_indices.items():
            for feat_idx in indices:
                if not self._driven[feat_idx]:
                    names.append(f"{interaction_type}:{self.features[feat_idx]}")
        return names

    def forward(self, bfi_diff: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Propagate BFI-TIPI differences through the causal DAG.

        Args:
            bfi_diff: Raw BFI-TIPI differences [batch_size, 5] (O, C, E, A, N)

        Returns:
            Dictionary mapping interaction_type -> predictions
            [batch_size, n_components] in normalized space
        """
        if bfi_diff.dim() != 2 or bfi_diff.shape[-1] != len(self.input_indices):
            raise ValueError(
                f"Expected bfi_diff of shape [batch, {len(self.input_indices)}], "
                f"got {tuple(bfi_diff.shape)}"
            )

        batch_size = bfi_diff.shape[0]
        normed_input = (bfi_diff - self.input_mean) / self.input_std

        # Per-feature column values, filled in causal order
        values: list[torch.Tensor | None] = [None] * self.n_features
        for i, idx in enumerate(self.input_indices):
            values[idx] = normed_input[:, i]

        masked_w = self.W * self.mask
        zeros = bfi_diff.new_zeros(batch_size)

        for v in self.causal_order:
            if values[v] is not None:
                continue  # exogenous input
            parents = self._parents[v]
            if not parents:
                values[v] = zeros
                continue
            col = zeros
            for j in parents:
                parent_value = values[j] if values[j] is not None else zeros
                col = col + masked_w[v, j] * parent_value
            values[v] = col

        x = torch.stack([v if v is not None else zeros for v in values], dim=1)

        return {
            interaction_type: x[:, indices]
            for interaction_type, indices in self.interaction_indices.items()
        }


def load_sem_head(
    model_data_path: str,
    norm_stats_path: str | None = None,
    interaction_columns: dict[str, list[str]] | None = None,
    trainable: bool = True,
) -> SEMHead:
    """
    Build a SEMHead from saved LiNGAM model data.

    Args:
        model_data_path: Path to a *_model_data.npy file containing a dict
            with 'adjacency_matrix', 'causal_order', and 'features'
        norm_stats_path: Path to the *_normalization_stats.csv file. If None,
            derived from model_data_path by filename convention
        interaction_columns: Mapping interaction_type -> column names. If
            None, loaded from configs.config.INTERACTION_TYPES
        trainable: Whether SEM coefficients are fine-tuned during training

    Returns:
        Initialized SEMHead
    """
    model_data_path = Path(model_data_path)
    if not model_data_path.exists():
        raise FileNotFoundError(f"Causal model data not found: {model_data_path}")

    model_data = np.load(model_data_path, allow_pickle=True).item()
    if not isinstance(model_data, dict) or "adjacency_matrix" not in model_data:
        raise ValueError(
            f"{model_data_path} does not contain model data (adjacency_matrix, "
            f"causal_order, features); expected a *_model_data.npy file"
        )
    for key in ("causal_order", "features"):
        if model_data.get(key) is None:
            raise ValueError(f"{model_data_path} is missing '{key}'")

    if norm_stats_path is None:
        norm_stats_path = (
            str(model_data_path)
            .replace("_model_data.npy", "_normalization_stats.csv")
            .replace("_robust_normalization_stats.csv", "_normalization_stats.csv")
        )
    norm_stats_path = Path(norm_stats_path)
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
    norm_stats = pd.read_csv(norm_stats_path)

    if interaction_columns is None:
        from configs.config import INTERACTION_TYPES

        interaction_columns = {name: cfg["columns"] for name, cfg in INTERACTION_TYPES.items()}

    return SEMHead(
        adjacency_matrix=model_data["adjacency_matrix"],
        features=list(model_data["features"]),
        causal_order=list(model_data["causal_order"]),
        interaction_columns=interaction_columns,
        trainable=trainable,
        norm_stats=norm_stats,
    )
