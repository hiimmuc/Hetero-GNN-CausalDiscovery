"""
Helper functions for data processing and utilities.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

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


def normalize(features: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize features using specified method.

    Args:
        features: Feature matrix to normalize
        method: Normalization method ('minmax', 'zscore')

    Returns:
        Normalized feature matrix
    """
    if method == "minmax":
        min_val = features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        normalized = (features - min_val) / (max_val - min_val + 1e-8)
    elif method == "zscore":
        mean_val = features.mean(axis=0, keepdims=True)
        std_val = features.std(axis=0, keepdims=True)
        normalized = (features - mean_val) / (std_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def normalize_features(
    features: np.ndarray,
    feature_names: List[str] = None,
    method: str = "mixed",
    tipi_scale: Tuple[float, float] = (1, 14),
    bfi_scale: Tuple[float, float] = (1, 10),
    print_stats: bool = True,
) -> np.ndarray:
    """
    Advanced feature normalization with different methods for different feature types.

    Args:
        features: Feature matrix [n_samples, n_features]
        feature_names: List of feature names
        method: Normalization method ('mixed', 'minmax', 'zscore')
        tipi_scale: Scale range for TIPI features
        bfi_scale: Scale range for BFI features
        print_stats: Whether to print normalization statistics

    Returns:
        Normalized feature matrix
    """
    if method != "mixed":
        # Use original normalization for non-mixed methods
        return normalize(features, method)

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
            normalized_values = (feature_values - min_val) / (max_val - min_val)
            norm_method = f"MinMax [{min_val}, {max_val}]"

        elif any(
            bfi_name in feature_name for bfi_name in ["BFI-O", "BFI-C", "BFI-E", "BFI-A", "BFI-N"]
        ):
            # BFI features: MinMax with (1, 10) scale
            min_val, max_val = bfi_scale
            normalized_values = (feature_values - min_val) / (max_val - min_val)
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

    # Create bipartite graph structure
    for user_id, item_id, rating in interactions:
        adj_mat[user_id, n_users + item_id] = rating
        adj_mat[n_users + item_id, user_id] = rating

    adj_mat = adj_mat.tolil()

    if adj_type == "plain":
        adj_mat = adj_mat.tocsr()
    elif adj_type == "norm":  # l2 normalization
        # Add self-loops and normalize
        adj_mat = adj_mat + sp.eye(n_nodes)
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocsr()
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
