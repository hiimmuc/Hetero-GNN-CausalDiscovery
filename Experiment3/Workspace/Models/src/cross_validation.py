"""
Cross-validation framework for GHCF models.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from configs.config import (
    CAUSAL_MODEL_PATH,
    DATA_DIR,
    DATABASE_PATH,
    EVAL_INTERVAL,
    NORMALIZE_INTERACTIONS,
    NUM_EPOCHS,
    RANDOM_SEED,
)
from sklearn.model_selection import KFold, StratifiedKFold
from train import Trainer
from utility.dataloader import eHMIDataset, load_processed_data
from utility.helper import load_interaction_norm_stats, set_seed


class CrossValidationSplitter:
    """Handle cross-validation data splitting."""

    def __init__(self, n_folds: int = 5, stratified: bool = True, random_state: int = RANDOM_SEED):
        """
        Initialize CV splitter.

        Args:
            n_folds: Number of folds for cross-validation
            stratified: Whether to use stratified splitting
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_state = random_state

        if stratified:
            self.splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
        else:
            self.splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def split_interactions(self, interactions: dict[str, list]) -> list[tuple[dict, dict]]:
        """
        Split interactions data into train/validation folds.

        Args:
            interactions: Dictionary of interaction types and their data

        Returns:
            List of (train_interactions, val_interactions) tuples for each fold
        """
        # Get primary interaction for stratification (use RATING if available)
        primary_interaction = "RATING"
        if primary_interaction not in interactions:
            primary_interaction = list(interactions.keys())[0]

        primary_data = interactions[primary_interaction]

        # Create user-based splits
        users = list(set([int(interaction[0]) for interaction in primary_data]))
        users = np.array(users)

        # For stratified splitting, create user activity levels
        if self.stratified:
            user_activity = {}
            for interaction in primary_data:
                user_id = int(interaction[0])
                user_activity[user_id] = user_activity.get(user_id, 0) + 1

            # Create activity level labels (low, medium, high)
            activity_values = np.array([user_activity.get(u, 0) for u in users])
            activity_labels = np.digitize(
                activity_values, np.percentile(activity_values, [33, 67])
            )

            splits = list(self.splitter.split(users, activity_labels))
        else:
            splits = list(self.splitter.split(users))

        fold_data = []
        for train_idx, val_idx in splits:
            train_users = set(users[train_idx])
            val_users = set(users[val_idx])

            # Split all interaction types based on user splits
            train_interactions = {}
            val_interactions = {}

            for interaction_type, interaction_list in interactions.items():
                train_interactions[interaction_type] = []
                val_interactions[interaction_type] = []

                for interaction in interaction_list:
                    user_id = int(interaction[0])
                    if user_id in train_users:
                        train_interactions[interaction_type].append(interaction)
                    elif user_id in val_users:
                        val_interactions[interaction_type].append(interaction)

            fold_data.append((train_interactions, val_interactions))

        return fold_data


class CrossValidationTrainer:
    """Cross-validation trainer for GHCF models."""

    def __init__(self, args, n_folds: int = 5, stratified: bool = True):
        """
        Initialize CV trainer.

        Args:
            args: Command line arguments
            n_folds: Number of folds for cross-validation
            stratified: Whether to use stratified splitting
        """
        self.args = args
        self.n_folds = n_folds
        self.stratified = stratified
        self.splitter = CrossValidationSplitter(n_folds, stratified)

        # Results storage
        self.fold_results = []
        self.cv_results = {}
        self.cv_dir = None

    def setup_cv_directory(self) -> None:
        """Setup cross-validation output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = getattr(self.args, "run_tag", "") or ""
        checkpoint_root = getattr(self.args, "checkpoint_root", "") or "checkpoints"
        dir_name = f"cv_{run_tag}_{timestamp}" if run_tag else f"cv_{timestamp}"
        self.cv_dir = Path(checkpoint_root) / dir_name
        self.cv_dir.mkdir(parents=True, exist_ok=True)

        # Save CV configuration
        cv_config = {
            "n_folds": self.n_folds,
            "stratified": self.stratified,
            "args": vars(self.args),
            "timestamp": timestamp,
        }

        with open(self.cv_dir / "cv_config.json", "w") as f:
            json.dump(cv_config, f, indent=2)

    def load_data(self) -> tuple[dict, dict, dict]:
        """Load dataset for cross-validation."""
        data_path = DATABASE_PATH
        interactions_hash = "_".join(sorted(self.args.interactions))
        # Scope the cache to the active data subset (SET) so 25/35 never collide.
        processed_data_dir = DATA_DIR / f"processed_{interactions_hash}"

        if processed_data_dir.exists() and (processed_data_dir / "metadata.pkl").exists():
            print(f"Loading processed data for interactions: {self.args.interactions}")
            train_data, test_data, metadata = load_processed_data(processed_data_dir)

            # Verify cached data matches requested interactions
            if set(metadata["interaction_types"]) != set(self.args.interactions):
                print("Cached interaction types don't match. Creating new dataset...")
                return self._create_new_dataset(data_path, processed_data_dir)
        else:
            print(f"Creating new dataset for interactions: {self.args.interactions}")
            return self._create_new_dataset(data_path, processed_data_dir)

        return train_data, test_data, metadata

    def _create_new_dataset(
        self, data_path: Path, processed_data_dir: Path
    ) -> tuple[dict, dict, dict]:
        """Create new dataset."""
        norm_stats = load_interaction_norm_stats(
            getattr(self.args, "causal_path", CAUSAL_MODEL_PATH)
        )
        dataset = eHMIDataset(
            data_path=str(data_path),
            interaction_types=self.args.interactions,
            test_size=0.0,  # Don't split here, we'll handle CV splitting
            adj_type=getattr(self.args, "adj_type", "plain"),
            random_state=RANDOM_SEED,
            normalize_interactions=getattr(
                self.args, "normalize_interactions", NORMALIZE_INTERACTIONS
            ),
            interaction_norm_stats=norm_stats,
        )

        # Get all data (no initial train/test split)
        all_data = dataset.get_train_data()  # This contains all data when test_size=0
        test_data = dataset.get_test_data()  # This will be empty

        metadata = {
            "n_users": dataset.n_users,
            "n_items": dataset.n_items,
            "interaction_types": dataset.interaction_types,
        }

        return all_data, test_data, metadata

    @staticmethod
    def _build_fold_adj_matrices(
        interactions: dict[str, list],
        n_users: int,
        n_items: int,
        adj_type: str,
    ) -> dict:
        """Build adjacency matrices from fold-specific training interactions only.

        Calling this per fold prevents validation-set edges from leaking into
        the GNN's message-passing graph (which would inflate performance metrics).

        Args:
            interactions: Fold's training interactions dict — values are lists of
                1D arrays ``[user_idx, item_idx, val0, ...]``.
            n_users: Total number of users (fixed across folds).
            n_items: Total number of items (fixed across folds).
            adj_type: Adjacency normalization type (plain / norm / mean).

        Returns:
            Dict mapping interaction_type → list of sparse torch tensors
            (one per interaction component).
        """
        from utility.helper import (
            create_adjacency_matrix,
            sparse_mx_to_torch_sparse_tensor,
        )

        adj_matrices: dict = {}
        for interaction_type, interaction_list in interactions.items():
            if not interaction_list:
                adj_matrices[interaction_type] = []
                continue

            arr = np.array(interaction_list)  # [n_rows, 2 + n_components]
            n_components = arr.shape[1] - 2
            matrices = []
            for comp_idx in range(n_components):
                comp_data = arr[:, [0, 1, comp_idx + 2]]
                adj = create_adjacency_matrix(comp_data, n_users, n_items, adj_type)
                matrices.append(sparse_mx_to_torch_sparse_tensor(adj))
            adj_matrices[interaction_type] = matrices

        return adj_matrices

    def run_fold(
        self, fold_idx: int, train_data: dict, val_data: dict, all_data: dict, metadata: dict
    ) -> dict:
        """
        Run training for a single fold.

        Args:
            fold_idx: Current fold index
            train_data: Training interactions for this fold
            val_data: Validation interactions for this fold
            all_data: All data (for features and adjacency matrices)
            metadata: Dataset metadata

        Returns:
            Fold results dictionary
        """
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*60}")

        # Create fold directory
        fold_dir = self.cv_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(exist_ok=True)

        # Prepare data for this fold.
        # Rebuild adjacency matrices from this fold's training interactions only
        # so that validation edges are not visible to the GNN during training.
        adj_type = getattr(self.args, "adj_type", "plain")
        fold_adj_matrices = self._build_fold_adj_matrices(
            train_data, metadata["n_users"], metadata["n_items"], adj_type
        )

        fold_train_data = {
            "interactions": train_data,
            "user_features": all_data["user_features"],
            "item_features": all_data["item_features"],
            "adj_matrices": fold_adj_matrices,
            "n_users": metadata["n_users"],
            "n_items": metadata["n_items"],
        }

        fold_val_data = {
            "interactions": val_data,
            "user_features": all_data["user_features"],
            "item_features": all_data["item_features"],
            # Validation pass uses the training graph (no val edges added)
            "adj_matrices": fold_adj_matrices,
            "n_users": metadata["n_users"],
            "n_items": metadata["n_items"],
        }

        # Create trainer for this fold
        fold_trainer = FoldTrainer(
            self.args, fold_train_data, fold_val_data, metadata, fold_dir, fold_idx
        )

        # Run training
        fold_results = fold_trainer.train()

        return fold_results

    def run(self) -> dict:
        """Run complete cross-validation."""
        set_seed(RANDOM_SEED)

        # Setup
        self.setup_cv_directory()

        # Load data
        print("Loading data for cross-validation...")
        all_data, _, metadata = self.load_data()

        # Create CV splits
        print(f"Creating {self.n_folds}-fold cross-validation splits...")
        fold_data = self.splitter.split_interactions(all_data["interactions"])

        print(f"Running {self.n_folds}-fold cross-validation...")
        cv_start_time = time.time()

        # Run each fold
        for fold_idx, (train_interactions, val_interactions) in enumerate(fold_data):
            fold_results = self.run_fold(
                fold_idx, train_interactions, val_interactions, all_data, metadata
            )
            self.fold_results.append(fold_results)

        # Aggregate results
        cv_time = time.time() - cv_start_time
        self.cv_results = self._aggregate_results(cv_time)

        # Save and print results
        self._save_cv_results()
        self._print_cv_results()

        return self.cv_results

    def _aggregate_results(self, cv_time: float) -> dict:
        """Aggregate results across all folds."""
        if not self.fold_results:
            return {}

        # Get all metric names from first fold
        metric_names = list(self.fold_results[0]["best_metrics"].keys())

        aggregated = {
            "cv_time": cv_time,
            "n_folds": self.n_folds,
            "fold_results": self.fold_results,
            "mean_metrics": {},
            "std_metrics": {},
            "best_fold": 0,
            "worst_fold": 0,
        }

        # Calculate mean and std for each metric
        for metric_name in metric_names:
            values = [fold["best_metrics"][metric_name] for fold in self.fold_results]
            aggregated["mean_metrics"][metric_name] = np.mean(values)
            aggregated["std_metrics"][metric_name] = np.std(values)

        # Find best and worst folds based on primary metric (recall@10)
        primary_metric = "recall@10"
        if primary_metric in metric_names:
            primary_values = [fold["best_metrics"][primary_metric] for fold in self.fold_results]
            aggregated["best_fold"] = int(np.argmax(primary_values))
            aggregated["worst_fold"] = int(np.argmin(primary_values))

        return aggregated

    def _save_cv_results(self) -> None:
        """Save cross-validation results."""
        with open(self.cv_dir / "cv_results.json", "w") as f:
            json.dump(self.cv_results, f, indent=2, default=str)

        # Save summary CSV
        self._save_cv_summary()

    def _save_cv_summary(self) -> None:
        """Save CV summary as CSV."""
        try:
            import pandas as pd

            # Create summary dataframe
            summary_data = []
            for i, fold_result in enumerate(self.fold_results):
                row = {"fold": i + 1}
                row.update(fold_result["best_metrics"])
                row["training_time"] = fold_result["training_time"]
                summary_data.append(row)

            # Add mean and std rows
            mean_row = {"fold": "mean"}
            std_row = {"fold": "std"}

            for metric_name in self.cv_results["mean_metrics"]:
                mean_row[metric_name] = self.cv_results["mean_metrics"][metric_name]
                std_row[metric_name] = self.cv_results["std_metrics"][metric_name]

            summary_data.extend([mean_row, std_row])

            df = pd.DataFrame(summary_data)
            df.to_csv(self.cv_dir / "cv_summary.csv", index=False)
        except ImportError:
            print("pandas not available, skipping CSV summary")

    def _print_cv_results(self) -> None:
        """Print cross-validation results."""
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*80}")

        print(f"Number of folds: {self.n_folds}")
        print(f"Total CV time: {self.cv_results['cv_time']:.2f}s")
        print(f"Best fold: {self.cv_results['best_fold'] + 1}")
        print(f"Worst fold: {self.cv_results['worst_fold'] + 1}")

        print("\nMean ± Std across folds:")
        print("-" * 40)
        for metric_name in self.cv_results["mean_metrics"]:
            mean_val = self.cv_results["mean_metrics"][metric_name]
            std_val = self.cv_results["std_metrics"][metric_name]
            print(f"{metric_name:15s}: {mean_val:.4f} ± {std_val:.4f}")

        print("\nPer-fold results:")
        print("-" * 40)
        for i, fold_result in enumerate(self.fold_results):
            print(f"Fold {i+1}:")
            for metric_name, value in fold_result["best_metrics"].items():
                print(f"  {metric_name:13s}: {value:.4f}")


class FoldTrainer(Trainer):
    """Trainer for a single CV fold."""

    def __init__(
        self, args, train_data: dict, val_data: dict, metadata: dict, fold_dir: Path, fold_idx: int
    ):
        """
        Initialize fold trainer.

        Args:
            args: Command line arguments
            train_data: Training data for this fold
            val_data: Validation data for this fold
            metadata: Dataset metadata
            fold_dir: Directory for this fold's outputs
            fold_idx: Fold index
        """
        # Initialize parent class
        super().__init__(args)

        # Override data
        self.train_data = train_data
        self.test_data = val_data  # Use validation data as test data
        self.n_users = metadata["n_users"]
        self.n_items = metadata["n_items"]
        self.interaction_types = metadata["interaction_types"]

        # Override directories
        self.model_dir = fold_dir
        self.fold_idx = fold_idx

        # Reduce epochs for CV (optional)
        self.max_epochs = getattr(args, "cv_epochs", NUM_EPOCHS // 2)

    def _setup_training_components(self) -> None:
        """Override to align the LR scheduler with this fold's actual epoch budget."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        super()._setup_training_components()
        # The base Trainer sets T_max = args.epoch (the full-run budget).
        # Rebuild the scheduler so it anneals over the shorter cv_epochs window.
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=self.args.lr * 0.01,
        )

    def setup(self) -> None:
        """Setup fold trainer (skip data loading)."""
        set_seed(RANDOM_SEED + self.fold_idx)  # Different seed per fold
        print(f"Using device: {self.device}")

        # Setup logging
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.model_dir / "logs")

        # Create model
        self._create_model()

        # Setup training components (calls overridden version above)
        self._setup_training_components()

    def train(self) -> dict:
        """Train model for this fold."""
        # Setup components
        self.setup()

        print(f"Starting fold {self.fold_idx + 1} training...")
        training_start_time = time.time()

        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()

            # Training
            train_loss, individual_losses, current_weights = self.train_epoch(epoch + 1)
            self.update_histories(train_loss, individual_losses)

            # Update learning rate scheduler
            self.scheduler.step()

            # Evaluation
            if epoch % EVAL_INTERVAL == 0 or epoch == self.max_epochs - 1:
                eval_metrics = self.evaluate_model(self.k_values)

                # Log metrics
                self.log_metrics(epoch, train_loss, individual_losses, eval_metrics)

                # Check for best model
                current_score = eval_metrics.get("recall@10", 0.0)
                is_best = current_score > self.best_score

                if is_best:
                    self.best_score = current_score
                    self.best_metrics = eval_metrics.copy()

                # Save checkpoint
                self.save_checkpoint(epoch, train_loss, eval_metrics, is_best)

                # Early stopping check
                if self.early_stopping(current_score):
                    print(f"Early stopping at epoch {epoch}")
                    break

                # Print progress
                epoch_time = time.time() - epoch_start_time
                if epoch % EVAL_INTERVAL == 0:  # Reduce printing frequency
                    self.print_progress(
                        epoch, train_loss, individual_losses, current_score, epoch_time
                    )

        # Final evaluation
        training_time = time.time() - training_start_time
        final_metrics = self.evaluate_model(self.k_values)

        # Save plots and results (mirrors Trainer.run())
        self.save_plots()
        self.save_results(training_time, final_metrics)
        self.writer.close()

        return {
            "fold_idx": self.fold_idx,
            "best_metrics": self.best_metrics,
            "final_metrics": final_metrics,
            "training_time": training_time,
            "best_score": self.best_score,
        }


def main():
    """Main cross-validation entry point (also supports plain training via --no-cv).

    Delegates argument parsing to ``utility.parser`` so all flags (--gnn_type,
    --n_heads, --cv, --causal, etc.) stay in one place.  The subcommand is
    synthesized as ``train`` so that ``parse_args`` works without a real
    subcommand token on the command line.
    """
    import sys

    # Inject the 'train' subcommand when this module is called directly so that
    # utility.parser.parse_args() (which expects a subcommand) works unchanged.
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "evaluate"):
        sys.argv.insert(1, "train")

    from utility.parser import parse_args

    args = parse_args()

    if getattr(args, "cv", False):
        cv_trainer = CrossValidationTrainer(args, n_folds=args.n_folds, stratified=args.stratified)
        cv_trainer.run()
    else:
        from train import Trainer

        trainer = Trainer(args)
        trainer.run()


if __name__ == "__main__":
    main()
