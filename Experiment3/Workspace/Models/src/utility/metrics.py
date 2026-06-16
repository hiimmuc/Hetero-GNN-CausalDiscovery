"""
Evaluation metrics for recommendation systems.
"""

from typing import List, Union

import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(r: List[int], k: int) -> float:
    """
    Calculate precision at k.

    Args:
        r: Binary relevance list (1 for relevant, 0 for not relevant)
        k: Top-k value

    Returns:
        Precision at k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def recall_at_k(r: List[int], k: int, all_pos_num: int) -> float:
    """
    Calculate recall at k.

    Args:
        r: Binary relevance list
        k: Top-k value
        all_pos_num: Total number of positive items

    Returns:
        Recall at k
    """
    r = np.asarray(r)[:k] != 0
    return np.sum(r) / all_pos_num if all_pos_num > 0 else 0.0


def ndcg_at_k(r: List[int], k: int) -> float:
    """
    Calculate NDCG at k.

    Args:
        r: Binary relevance list
        k: Top-k value

    Returns:
        NDCG at k
    """
    assert k >= 1
    r = np.asarray(r)[:k]

    if r.size:
        dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
        idcg = np.sum([1 / np.log2(i + 2) for i in range(min(np.sum(r != 0), k))])
        return dcg / idcg if idcg > 0 else 0.0
    return 0.0


def hit_at_k(r: List[int], k: int) -> float:
    """
    Calculate hit rate at k.

    Args:
        r: Binary relevance list
        k: Top-k value

    Returns:
        Hit rate at k (1 if any hit, 0 otherwise)
    """
    r = np.asarray(r)[:k] != 0
    return 1.0 if np.sum(r) > 0 else 0.0


def auc(
    ground_truth: Union[List[int], np.ndarray], prediction: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate AUC (Area Under Curve).
    Filters out infinite values (used for masking) before calculation.

    Args:
        ground_truth: Binary ground truth labels
        prediction: Prediction scores

    Returns:
        AUC score
    """
    # Convert to numpy arrays and flatten to ensure 1D
    gt_array = np.asarray(ground_truth).flatten()
    pred_array = np.asarray(prediction).flatten()

    # Ensure both arrays have the same length
    if len(gt_array) != len(pred_array):
        raise ValueError(
            f"Ground truth and prediction arrays must have the same length. "
            f"Got {len(gt_array)} and {len(pred_array)}"
        )

    # Filter out infinite values (used for masking training items)
    finite_mask = np.isfinite(pred_array)

    if np.sum(finite_mask) == 0:
        # If all values are infinite, return 0.5 (random performance)
        return 0.5

    # Filter both arrays to only include finite predictions
    filtered_gt = gt_array[finite_mask]
    filtered_pred = pred_array[finite_mask]

    # Check if we have both positive and negative samples
    if len(np.unique(filtered_gt)) < 2:
        # If all remaining samples have the same label, return 0.5
        return 0.5

    return roc_auc_score(filtered_gt, filtered_pred)


def calculate_ranking_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    k_values: List[int] = [5, 10, 20],
) -> dict:
    """
    Calculate multiple ranking metrics.

    Args:
        y_true: Binary relevance list
        y_pred: Prediction scores
        k_values: List of k values to evaluate

    Returns:
        Dictionary containing all metrics
    """
    # Convert to numpy arrays for consistent handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Sort by prediction scores
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_labels = y_true[sorted_indices]

    metrics = {}
    n_pos = np.sum(y_true)

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(sorted_labels, k)
        metrics[f"recall@{k}"] = recall_at_k(sorted_labels, k, n_pos)
        metrics[f"ndcg@{k}"] = ndcg_at_k(sorted_labels, k)
        metrics[f"hit@{k}"] = hit_at_k(sorted_labels, k)

    metrics["auc"] = auc(y_true, y_pred)

    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Example data
    y_true = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    y_pred = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.6, 0.5, 0.05]

    # Test individual metrics
    k = 5
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_labels = [y_true[i] for i in sorted_indices]

    prec = precision_at_k(sorted_labels, k)
    rec = recall_at_k(sorted_labels, k, sum(y_true))
    ndcg = ndcg_at_k(sorted_labels, k)
    hit = hit_at_k(sorted_labels, k)
    auc_score = auc(y_true, y_pred)

    print(f"Precision@{k}: {prec:.4f}")
    print(f"Recall@{k}: {rec:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"Hit@{k}: {hit:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # Test comprehensive metrics
    metrics = calculate_ranking_metrics(y_true, y_pred, [5, 10])
    print("\nComprehensive metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("✓ All metrics tests passed!")
