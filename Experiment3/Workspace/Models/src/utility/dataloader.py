"""
Data loader for eHMI recommendation system with heterogeneous multi-relational interactions.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from helper import (
        create_adjacency_matrix,
        encode_categorical_features,
        normalize_features,
        sparse_mx_to_torch_sparse_tensor,
    )
except Exception:
    from utility.helper import (
        create_adjacency_matrix,
        encode_categorical_features,
        normalize_features,
        sparse_mx_to_torch_sparse_tensor,
    )


class eHMIDataset(Dataset):
    """
    Dataset class for eHMI recommendation with heterogeneous interactions.
    """

    def __init__(
        self,
        data_path: str,
        interaction_types: List[str],
        test_size: float = 0.2,
        adj_type: str = "norm",
        normalize_interactions: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize eHMI dataset.

        Args:
            data_path: Path to the database file
            interaction_types: List of interaction types to use
            test_size: Ratio for test split
            adj_type: Type of adjacency matrix normalization
            normalize_interactions: Whether to normalize interaction values
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.interaction_types = interaction_types
        self.test_size = test_size
        self.adj_type = adj_type
        self.normalize_interactions = normalize_interactions
        self.random_state = random_state

        # Load and process data
        self._load_and_process_data()

    def _load_and_process_data(self) -> None:
        """Load and process the raw data from database.xlsx"""
        print("Loading data from database.xlsx...")

        # Load the main database
        df = pd.read_excel(self.data_path)
        print(f"Loaded {len(df)} interactions")

        # Define feature columns based on your specification
        self.user_feature_cols = ["Participant ID", "Participant Gender"] + [
            "TIPI-O",
            "TIPI-C",
            "TIPI-E",
            "TIPI-A",
            "TIPI-N",
        ]
        self.item_feature_cols = ["eHMI ID", "eHMI Gender", "eHMI Content Type", "eHMI Character"]

        # Define interaction columns based on interaction types
        self.interaction_columns = self._get_interaction_columns()

        # Create user and item mappings
        unique_users = df["Participant ID"].unique()
        unique_items = df["eHMI ID"].unique()

        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        self.n_nodes = self.n_users + self.n_items

        print(f"Number of users: {self.n_users}")
        print(f"Number of items: {self.n_items}")

        # Split data first before processing
        self._split_train_test_by_user(df)

        # Process features and interactions for each split
        self._process_split_data()

    def _get_interaction_columns(self) -> Dict[str, List[str]]:
        """Get interaction columns based on selected interaction types."""
        all_interactions = {
            "RATING": ["Rating score"],
            "BFI_PERCEPTION": ["BFI-O", "BFI-C", "BFI-E", "BFI-A", "BFI-N"],
            "VOICE_QUALITY": ["MOS-X2 Q1", "MOS-X2 Q2", "MOS-X2 Q3", "MOS-X2 Q4"],
            "SUBJECTIVE_FEELING": ["SQ1", "SQ2", "SQ3", "SQ4", "SQ5"],
        }

        interaction_columns = {}
        for interaction_type in self.interaction_types:
            if interaction_type in all_interactions:
                interaction_columns[interaction_type] = all_interactions[interaction_type]

        return interaction_columns

    def _split_train_test_by_user(self, df: pd.DataFrame) -> None:
        """Split data by taking last test_ratio of records for each user as test."""
        print(f"Splitting data by user (test ratio: {self.test_size})...")
        
        self.train_df = []
        self.test_df = []
        
        for user_id in df["Participant ID"].unique():
            user_data = df[df["Participant ID"] == user_id].copy()
            
            # Sort by some order (you might want to add a timestamp column if available)
            # For now, we'll use the original order
            user_data = user_data.reset_index(drop=True)
            
            n_interactions = len(user_data)
            n_test = int(n_interactions * self.test_size)
            if self.test_size > 0:
                n_test = max(1, n_test)  # At least 1 test sample when splitting
            n_train = n_interactions - n_test
            
            # Take last n_test records as test
            train_data = user_data.iloc[:n_train]
            test_data = user_data.iloc[n_train:]
            
            self.train_df.append(train_data)
            self.test_df.append(test_data)
            
            print(f"User {user_id}: {n_train} train, {n_test} test interactions")
        
        # Combine all users
        self.train_df = pd.concat(self.train_df, ignore_index=True)
        self.test_df = pd.concat(self.test_df, ignore_index=True)
        
        print(f"Total - Train: {len(self.train_df)}, Test: {len(self.test_df)}")

    def _process_split_data(self) -> None:
        """Process features and interactions for train and test sets separately."""
        
        # Process node features (these will be the same for both sets)
        self._process_node_features_combined()
        
        # Process interactions and adjacency matrices for train set
        print("\nProcessing training data...")
        self.train_interactions, self.train_adj_matrices = self._process_interactions_for_split(
            self.train_df, "train"
        )
        
        # Process interactions for test set (no adjacency matrices needed for test)
        print("\nProcessing test data...")
        self.test_interactions, _ = self._process_interactions_for_split(
            self.test_df, "test", create_adj_matrices=False
        )

    def _process_node_features_combined(self) -> None:
        """Process user and item features using complete dataset."""
        print("Processing node features...")
        
        # Combine train and test for complete feature processing
        combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        # Process user features
        user_df = combined_df[self.user_feature_cols].drop_duplicates()
        user_df = (
            user_df.set_index("Participant ID")
            .reindex([self.idx_to_user[i] for i in range(self.n_users)])
            .reset_index()
        )

        # Encode categorical features for users
        user_cat_cols = ["Participant Gender"]
        user_df_encoded, self.user_encodings = encode_categorical_features(user_df, user_cat_cols)

        # Extract numerical features for users
        user_num_cols = ["TIPI-O", "TIPI-C", "TIPI-E", "TIPI-A", "TIPI-N"]
        user_features = user_df_encoded[user_cat_cols + user_num_cols].values.astype(np.float32)

        # Import scales from config
        from configs.config import BFI_SCALE, TIPI_SCALE

        # Normalize user features with advanced normalization
        feature_names = user_cat_cols + user_num_cols
        self.user_features = normalize_features(
            user_features,
            feature_names,
            method="mixed",
            tipi_scale=TIPI_SCALE,
            bfi_scale=BFI_SCALE,
            print_stats=True,
        )

        # Process item features
        item_df = combined_df[self.item_feature_cols].drop_duplicates()
        item_df = (
            item_df.set_index("eHMI ID")
            .reindex([self.idx_to_item[i] for i in range(self.n_items)])
            .reset_index()
        )

        # Encode categorical features for items
        item_cat_cols = ["eHMI Gender", "eHMI Content Type", "eHMI Character"]
        item_df_encoded, self.item_encodings = encode_categorical_features(item_df, item_cat_cols)

        # Extract features for items (all categorical in this case)
        item_features = item_df_encoded[item_cat_cols].values.astype(np.float32)
        self.item_features = item_features  # No normalization needed for categorical features

        print(f"User feature shape: {self.user_features.shape}")
        print(f"Item feature shape: {self.item_features.shape}")

    def _process_interactions_for_split(
        self, df: pd.DataFrame, split_name: str, create_adj_matrices: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List]]:
        """Process interactions for a specific data split."""
        print(f"Processing interactions for {split_name} set...")

        interactions = {}
        adj_matrices = {} if create_adj_matrices else None

        for interaction_type, columns in self.interaction_columns.items():
            print(f"Processing {interaction_type} for {split_name}...")

            # Process interactions for this specific type
            interactions_list = []

            for _, row in df.iterrows():
                user_idx = self.user_to_idx[row["Participant ID"]]
                item_idx = self.item_to_idx[row["eHMI ID"]]

                # Collect all values for this interaction type
                values = []
                all_valid = True

                for col in columns:
                    rating = row[col]
                    if pd.isna(rating):
                        all_valid = False
                        break
                    values.append(rating)

                # Only add interaction if all values are valid
                if all_valid:
                    interaction_row = [user_idx, item_idx] + values
                    interactions_list.append(interaction_row)

            # Convert to numpy array for this interaction type
            interactions_array = (
                np.array(interactions_list)
                if interactions_list
                else np.empty((0, len(columns) + 2))
            )

            # Normalize interaction values if we have interactions and normalization is enabled
            if len(interactions_array) > 0 and self.normalize_interactions:
                # Use the same normalization logic as before
                from configs.config import BFI_SCALE

                interaction_values = interactions_array[:, 2:]

                if interaction_type in ["BFI_PERCEPTION"]:
                    min_val, max_val = BFI_SCALE
                    normalized_values = (interaction_values - min_val) / (max_val - min_val)
                    norm_method = f"MinMax with BFI scale [{min_val}, {max_val}]"
                elif interaction_type in ["VOICE_QUALITY", "SUBJECTIVE_FEELING", "RATING"]:
                    mean_vals = np.mean(interaction_values, axis=0)
                    std_vals = np.std(interaction_values, axis=0)
                    normalized_values = (interaction_values - mean_vals) / (std_vals + 1e-8)
                    norm_method = "Z-Score"
                else:
                    mean_vals = np.mean(interaction_values, axis=0)
                    std_vals = np.std(interaction_values, axis=0)
                    normalized_values = (interaction_values - mean_vals) / (std_vals + 1e-8)
                    norm_method = "Z-Score (default)"

                interactions_array[:, 2:] = normalized_values
                print(f"Normalized {split_name} {interaction_type} using {norm_method}")

            interactions[interaction_type] = interactions_array

            # Create adjacency matrices only for training set
            if create_adj_matrices and len(interactions_array) > 0:
                matrices_for_current_type = []
                num_components = len(columns)

                for comp_idx in range(num_components):
                    component_interactions = interactions_array[:, [0, 1, comp_idx + 2]]
                    adj_matrix = create_adjacency_matrix(
                        component_interactions, self.n_users, self.n_items, self.adj_type
                    )
                    sparse_tensor = sparse_mx_to_torch_sparse_tensor(adj_matrix)
                    matrices_for_current_type.append(sparse_tensor)

                adj_matrices[interaction_type] = matrices_for_current_type
            elif create_adj_matrices:
                adj_matrices[interaction_type] = []

            print(f"{split_name} {interaction_type}: {len(interactions_array)} interactions")

        return interactions, adj_matrices

    def get_train_data(self) -> Dict[str, Any]:
        """Get training data."""
        return {
            "interactions": self.train_interactions,
            "adj_matrices": self.train_adj_matrices,
            "user_features": torch.FloatTensor(self.user_features),
            "item_features": torch.FloatTensor(self.item_features),
            "n_users": self.n_users,
            "n_items": self.n_items,
        }

    def get_test_data(self) -> Dict[str, Any]:
        """Get test data."""
        return {
            "interactions": self.test_interactions,
            "user_features": torch.FloatTensor(self.user_features),
            "item_features": torch.FloatTensor(self.item_features),
            "n_users": self.n_users,
            "n_items": self.n_items,
        }

    def save_processed_data(self, save_dir: str) -> None:
        """Save processed data to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save train/test data
        with open(save_path / "train_data.pkl", "wb") as f:
            pickle.dump(self.get_train_data(), f)

        with open(save_path / "test_data.pkl", "wb") as f:
            pickle.dump(self.get_test_data(), f)

        # Save metadata
        metadata = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "interaction_types": self.interaction_types,
            "user_encodings": self.user_encodings,
            "item_encodings": self.item_encodings,
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
        }

        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"Data saved to {save_path}")


def load_processed_data(data_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load processed data from disk.

    Args:
        data_dir: Directory containing processed data

    Returns:
        Tuple of (train_data, test_data, metadata)
    """
    data_path = Path(data_dir)

    with open(data_path / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    with open(data_path / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    with open(data_path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return train_data, test_data, metadata


if __name__ == "__main__":
    # Test data loader
    print("Testing eHMI Dataset...")

    # Test with sample data path (adjust path as needed)
    data_path = "../data/database.xlsx"

    if os.path.exists(data_path):
        try:
            # Test original dataset creation
            dataset = eHMIDataset(
                data_path=data_path,
                interaction_types=[
                    "RATING",
                    "BFI_PERCEPTION",
                ],
                test_size=0.2,
                normalize_interactions=True,
            )

            # Get data
            train_data = dataset.get_train_data()
            test_data = dataset.get_test_data()

            print("✓ Original Dataset created successfully")
            print(f"  - Users: {train_data['n_users']}")
            print(f"  - Items: {train_data['n_items']}")
            print(f"  - User features shape: {train_data['user_features'].shape}")
            print(f"  - Item features shape: {train_data['item_features'].shape}")
            print(f"  - Interactions: {list(train_data['interactions'].keys())}")
            print(f"  - Adjacency matrices: {list(train_data['adj_matrices'].keys())}")

            # Test saving and loading
            dataset.save_processed_data("./test_data")
            loaded_train, loaded_test, metadata = load_processed_data("./test_data")

            print("✓ Original data saving and loading works")
        except Exception as e:
            print(f"Error testing with real data: {e}")
            print("This is expected if database.xlsx is not available")
    else:
        print("database.xlsx not found, skipping real data test")

    print("✓ Data loader module ready")
