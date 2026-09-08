"""
Helper functions for data processing and utilities.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minmax_scale(values, scale_min: float, scale_max: float):
    """Rescale ``values`` from a fixed [scale_min, scale_max] range to [0, 1].

    Unlike a distribution-based MinMax normalizer, the bounds are the
    questionnaire's declared range (e.g. BFI_SCALE, TIPI_SCALE), not
    ``values.min()``/``values.max()``. Works on numpy arrays and pandas
    Series alike.
    """
    return (values - scale_min) / (scale_max - scale_min)


def normalize_features(
    features: np.ndarray,
    feature_names: List[str] = None,
    tipi_scale: Tuple[float, float] = (1, 14),
    bfi_scale: Tuple[float, float] = (1, 10),
    print_stats: bool = True,
) -> np.ndarray:
    """
    Advanced feature normalization with different methods for different feature types.

    Args:
        features: Feature matrix [n_samples, n_features]
        feature_names: List of feature names
        tipi_scale: Scale range for TIPI features
        bfi_scale: Scale range for BFI features
        print_stats: Whether to print normalization statistics

    Returns:
        Normalized feature matrix
    """
    normalized_features = features.copy()

    if print_stats:
        print("\n" + "=" * 80)
        print("FEATURE NORMALIZATION STATISTICS")
        print("=" * 80)

    for i, feature_name in enumerate(feature_names):
        feature_values = features[:, i]

        # Determine normalization method based on feature type
        if any(
            tipi_name in feature_name
            for tipi_name in ["TIPI-O", "TIPI-C", "TIPI-E", "TIPI-A", "TIPI-N"]
        ):
            # TIPI features: MinMax with (1, 14) scale
            min_val, max_val = tipi_scale
            normalized_values = minmax_scale(feature_values, min_val, max_val)
            norm_method = f"MinMax [{min_val}, {max_val}]"

        elif any(
            bfi_name in feature_name for bfi_name in ["BFI-O", "BFI-C", "BFI-E", "BFI-A", "BFI-N"]
        ):
            # BFI features: MinMax with (1, 10) scale
            min_val, max_val = bfi_scale
            normalized_values = minmax_scale(feature_values, min_val, max_val)
            norm_method = f"MinMax [{min_val}, {max_val}]"

        elif any(
            interaction_name in feature_name for interaction_name in ["MOS-X2", "SQ", "Rating"]
        ):
            # Interaction features: Z-score normalization
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            normalized_values = (feature_values - mean_val) / (std_val + 1e-8)
            norm_method = f"Z-Score (μ={mean_val:.3f}, σ={std_val:.3f})"

        else:
            # Default: MinMax for other features (like gender encoding)
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            if max_val - min_val > 1e-8:
                normalized_values = (feature_values - min_val) / (max_val - min_val)
                norm_method = f"MinMax [{min_val:.3f}, {max_val:.3f}]"
            else:
                normalized_values = feature_values
                norm_method = "No normalization (constant)"

        normalized_features[:, i] = normalized_values

        if print_stats:
            print(
                f"{feature_name:20s} | {norm_method:25s} | "
                f"Original: [{np.min(feature_values):.3f}, {np.max(feature_values):.3f}] | "
                f"Normalized: [{np.min(normalized_values):.3f}, {np.max(normalized_values):.3f}]"
            )

    if print_stats:
        print("=" * 80)
        print(f"Normalized {len(feature_names)} features using mixed normalization strategy")
        print("=" * 80)

    return normalized_features


def create_adjacency_matrix(
    interactions: np.ndarray, n_users: int, n_items: int, adj_type: str = "norm"
) -> sp.csr_matrix:
    """
    Create adjacency matrix from user-item interactions.

    Args:
        interactions: Array of (user_id, item_id, rating) tuples
        n_users: Number of users
        n_items: Number of items
        adj_type: Type of adjacency matrix ('plain', 'norm', 'gcmc')

    Returns:
        Adjacency matrix in CSR format
    """
    n_nodes = n_users + n_items
    adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)

    # Create bipartite graph structure. Cast the index columns explicitly: when
    # the interaction rows carry float values (e.g. fold-local re-normalized
    # targets) the whole row is float64, and scipy rejects float indices even
    # when integral-valued.
    for user_id, item_id, rating in interactions:
        u, it = int(user_id), int(item_id)
        adj_mat[u, n_users + it] = rating
        adj_mat[n_users + it, u] = rating

    adj_mat = adj_mat.tolil()

    if adj_type == "plain":
        adj_mat = adj_mat.tocsr()
    elif adj_type == "pre":
        # GHCF/LightGCN symmetric normalization WITHOUT self-loops:
        #   A_hat = D^-1/2 @ A @ D^-1/2   (no + I)
        # This matches the reference GHCF's `pre_adj_mat`. Self information is
        # supplied separately by the layer's self-transform (and the layer-0
        # term in the mean over layers), so adding a self-loop here would
        # double-count it.
        #
        # GHCF's edges are binary, so its degree equals the interaction count.
        # Here the edge weights are signed (z-scored) interaction values, so the
        # degree is taken from the *structure* (number of incident edges), not
        # the signed weight sum -- otherwise D^-1/2 would hit negative/zero row
        # sums and produce NaN. This reduces exactly to GHCF for binary weights.
        adj_csr = adj_mat.tocsr()
        degree = np.asarray((adj_csr != 0).sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        nz = degree > 0
        d_inv_sqrt[nz] = np.power(degree[nz].astype(np.float32), -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_mat = d_mat_inv_sqrt.dot(adj_csr).dot(d_mat_inv_sqrt).tocsr()
    elif adj_type == "norm":  # symmetric normalization with self-loops
        # Add self-loops and normalize. Degree is taken from edge *structure*
        # (count of nonzero entries per row), not the signed weight sum: RATING
        # weights are z-scored and can be negative, so summing them can produce
        # a negative row sum, and D^-1/2 of a negative number is NaN (unlike
        # D^-1 in the 'mean' branch below, which stays finite for negative
        # sums). Mirrors the 'pre' branch's fix for the same signed-weight issue.
        adj_csr = (adj_mat + sp.eye(n_nodes)).tocsr()
        degree = np.asarray((adj_csr != 0).sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        nz = degree > 0
        d_inv_sqrt[nz] = np.power(degree[nz].astype(np.float32), -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_mat = d_mat_inv_sqrt.dot(adj_csr).dot(d_mat_inv_sqrt).tocsr()
    elif adj_type == "mean":
        # Add self-loops and normalize by mean
        adj_mat = adj_mat + sp.eye(n_nodes)
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_mean = np.power(rowsum, -1.0)
        d_inv_mean[np.isinf(d_inv_mean)] = 0.0
        d_mat_inv_mean = sp.diags(d_inv_mean)
        adj_mat = d_mat_inv_mean.dot(adj_mat).tocsr()

    return adj_mat


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.csr_matrix) -> torch.sparse.FloatTensor:
    """
    Convert scipy sparse matrix to PyTorch sparse tensor.

    Args:
        sparse_mx: Scipy sparse matrix

    Returns:
        PyTorch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)


def encode_categorical_features(
    data: pd.DataFrame, cat_columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical features using label encoding.

    Args:
        data: DataFrame with categorical features
        cat_columns: List of categorical column names

    Returns:
        Tuple of (encoded_data, encoding_maps)
    """
    data_encoded = data.copy()
    encoding_maps = {}

    for col in cat_columns:
        if col in data_encoded.columns:
            unique_values = data_encoded[col].unique()
            encoding_map = {val: idx for idx, val in enumerate(unique_values)}
            data_encoded[col] = data_encoded[col].map(encoding_map)
            encoding_maps[col] = encoding_map

    return data_encoded, encoding_maps


def resolve_causal_norm_stats_path(causal_model_path: str) -> Path:
    """Resolve the normalization-stats CSV that accompanies a saved causal model.

    Mirrors the suffix convention used when the LiNGAM model is saved (and the
    one ``CausalMixin``/``SEMHead`` use to locate the same file), so the
    dataloader and the SEM head read identical mean/std values, e.g.
    ``..._robust_model_data.npy`` -> ``..._normalization_stats.csv``.
    """
    name = str(causal_model_path)
    for suffix in (
        "_robust_model_data.npy",
        "_model_data.npy",
        "_robust_adjacency_matrix.npy",
        "_adjacency_matrix.npy",
    ):
        if name.endswith(suffix):
            return Path(name[: -len(suffix)] + "_normalization_stats.csv")
    # Last resort: a sibling file derived from the stem
    p = Path(name)
    return p.with_name(p.stem + "_normalization_stats.csv")


def load_interaction_norm_stats(causal_model_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load interaction normalization stats (per-feature mean/std) from the
    causal model's stats CSV.

    These are the *same* statistics the SEM head uses, so z-scoring the
    RATING / VOICE_QUALITY / SUBJECTIVE_FEELING targets with them places the
    targets in the SEM head's output space exactly.

    Returns the stats DataFrame, or ``None`` if the path is unset/missing (the
    dataloader then falls back to per-split z-scoring).
    """
    if not causal_model_path:
        return None
    stats_path = resolve_causal_norm_stats_path(causal_model_path)
    if not stats_path.exists():
        print(
            f"[norm] Causal normalization stats not found at {stats_path}; "
            f"interaction targets will fall back to per-split z-scoring."
        )
        return None
    print(f"[norm] Using causal normalization stats for interaction targets: {stats_path}")
    return pd.read_csv(stats_path)


def resolve_causal_paths(causal_model_path: str, causal_config_path: str) -> Tuple[str, str]:
    """Validate causal model paths and return them.

    Raises FileNotFoundError if either path does not exist.
    """
    if not Path(causal_model_path).exists():
        raise FileNotFoundError(f"Causal model file not found: {causal_model_path}")
    if not Path(causal_config_path).exists():
        raise FileNotFoundError(f"Causal config file not found: {causal_config_path}")

    print(f"Causal mode enabled. Model path: {causal_model_path}")
    print(f"Causal config path: {causal_config_path}")

    return causal_model_path, causal_config_path
