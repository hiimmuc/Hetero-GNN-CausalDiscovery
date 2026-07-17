"""
Batch processing utilities for evaluation and training.
"""

from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

try:
    from configs.config import RATING_THRESHOLD
    from metrics import calculate_ranking_metrics
except Exception:
    from configs.config import RATING_THRESHOLD

    from utility.metrics import calculate_ranking_metrics


class BatchProcessor:
    """
    Handles batch processing for training and evaluation.
    """

    def __init__(self, n_users: int, n_items: int, batch_size: int = 64):
        """
        Initialize batch processor.

        Args:
            n_users: Number of users
            n_items: Number of items
            batch_size: Batch size for processing
        """
        self.n_users = n_users
        self.n_items = n_items
        self.batch_size = batch_size

    def create_training_batches(
        self, interactions: Dict[str, np.ndarray], batch_size: int = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create training batches from interactions.

        Args:
            interactions: Dictionary of interaction arrays for each type
            batch_size: Batch size override

        Returns:
            List of batch dictionaries
        """
        batch_size = batch_size or self.batch_size
        batches = []

        # Process each interaction type separately to maintain proper dimensionality
        all_batch_data = []

        for interaction_type, interaction_data in interactions.items():
            if len(interaction_data) == 0:
                continue

            # interaction_data shape: [n_interactions, 2 + n_values]
            # where first 2 columns are user_idx, item_idx, and rest are values
            n_interactions = len(interaction_data)
            for i in range(n_interactions):
                interaction_row = interaction_data[i]
                user_idx = int(interaction_row[0])
                item_idx = int(interaction_row[1])
                values = interaction_row[2:]  # Multi-dimensional values

                all_batch_data.append(
                    {
                        "user": user_idx,
                        "item": item_idx,
                        "values": values,
                        "interaction_type": interaction_type,
                    }
                )

        # Shuffle and create batches
        np.random.shuffle(all_batch_data)
        n_total = len(all_batch_data)

        # Create batches
        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            batch_data_slice = all_batch_data[start_idx:end_idx]

            # Group by interaction type within batch
            users = []
            items = []
            values_list = []
            interaction_types = []

            for data_point in batch_data_slice:
                users.append(data_point["user"])
                items.append(data_point["item"])
                values_list.append(data_point["values"])
                interaction_types.append(data_point["interaction_type"])

            batch = {
                "users": torch.LongTensor(users),
                "items": torch.LongTensor(items),
                "values": values_list,  # Keep as list since dimensions can vary
                "interaction_types": interaction_types,
            }
            batches.append(batch)

        return batches

    @staticmethod
    def create_train_sets(
        train_interactions: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[int, set]]:
        """Create per-user training item sets for exclusion during evaluation."""
        train_sets = {}
        for interaction_type, interactions in train_interactions.items():
            train_sets[interaction_type] = {}
            for interaction_row in interactions:
                if len(interaction_row) < 2:
                    continue
                user_id, item_id = int(interaction_row[0]), int(interaction_row[1])
                if user_id not in train_sets[interaction_type]:
                    train_sets[interaction_type][user_id] = set()
                train_sets[interaction_type][user_id].add(item_id)
        return train_sets

    @staticmethod
    def get_test_users(test_interactions: Dict[str, np.ndarray]) -> List[int]:
        """Get the set of users appearing in any test interaction."""
        test_users = set()
        for interactions in test_interactions.values():
            for interaction_row in interactions:
                if len(interaction_row) >= 2:
                    test_users.add(int(interaction_row[0]))
        return sorted(test_users)

    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_interactions: Dict[str, np.ndarray],
        train_interactions: Dict[str, np.ndarray],
        k_values: List[int] = [5, 10, 20],
        device: str = "cpu",
        train_data: Dict = None,
        target_interaction: str = "RATING",
        show_progress: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model ranking performance on test data.

        This is the single ranking-evaluation implementation, used both during
        training (Trainer) and offline (Evaluator).

        Args:
            model: Trained model
            test_interactions: Test interactions
            train_interactions: Training interactions (for exclusion)
            k_values: List of k values for ranking metrics
            device: Device to run evaluation on
            train_data: Full training data dictionary (needed for GHCF model)
            target_interaction: Primary interaction type for recommendation
            show_progress: Whether to show a progress bar over test users

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_metrics = []

        train_sets = self.create_train_sets(train_interactions)
        test_users = self.get_test_users(test_interactions)
        users_iter = tqdm(test_users) if show_progress else test_users

        with torch.no_grad():
            for user_id in users_iter:
                # Predict scores for all items for this user
                user_tensor = torch.LongTensor([user_id]).repeat(self.n_items).to(device)
                item_tensor = torch.arange(self.n_items, dtype=torch.long).to(device)

                if hasattr(model, "predict") and train_data is not None:
                    # GHCF model requires features and adjacency matrices
                    predictions = model.predict(
                        user_tensor,
                        item_tensor,
                        train_data["user_features"].to(device),
                        train_data["item_features"].to(device),
                        {
                            k: [matrix.to(device) for matrix in matrices]
                            for k, matrices in train_data["adj_matrices"].items()
                        },
                        target_interaction,
                    )
                else:
                    # Fallback for other model types
                    predictions = model(user_tensor, item_tensor)

                predictions = predictions.cpu().numpy().flatten()

                # Create ground truth labels from the test set
                ground_truth = np.zeros(self.n_items)
                for interaction_row in test_interactions.get(target_interaction, []):
                    if len(interaction_row) >= 3:
                        u_id, i_id = int(interaction_row[0]), int(interaction_row[1])
                        rating = float(interaction_row[2])
                        if u_id == user_id and rating >= RATING_THRESHOLD:
                            ground_truth[i_id] = 1
                    elif len(interaction_row) >= 2:
                        # No rating value available: treat interaction as positive
                        u_id, i_id = int(interaction_row[0]), int(interaction_row[1])
                        if u_id == user_id:
                            ground_truth[i_id] = 1

                # Exclude training items from predictions
                for train_item in train_sets.get(target_interaction, {}).get(user_id, ()):
                    predictions[train_item] = float("-inf")

                # Calculate metrics if there are positive items
                if np.sum(ground_truth) > 0:
                    user_metrics = calculate_ranking_metrics(
                        ground_truth.tolist(), predictions.tolist(), k_values
                    )
                    all_metrics.append(user_metrics)

        # Average metrics across all users with positive items
        if all_metrics:
            return {
                key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()
            }
        return {f"precision@{k}": 0.0 for k in k_values}
