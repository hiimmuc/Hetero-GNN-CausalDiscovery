"""
Cross-validation framework for GHCF models.
"""

import copy
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from configs.config import (
    CAUSAL_CONFIG_PATH,
    CAUSAL_MODEL_PATH,
    CV_CAUSAL_JOBS,
    DATA_DIR,
    DATABASE_PATH,
    EVAL_INTERVAL,
    NORMALIZE_INTERACTIONS,
    NUM_EPOCHS,
    RANDOM_SEED,
)
from sklearn.model_selection import KFold, StratifiedKFold
from train import Trainer
from utility.cv_causal import (
    NORM_STATS_CSV,
    ROBUST_NPY,
    assert_split_matches,
    causal_fold_key,
    ensure_fold_artifact,
    norm_lookup_from_csv,
    read_folds_manifest,
    refold_normalize_interactions,
    write_folds_manifest,
)
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

    @staticmethod
    def _units_and_activity(unit_occurrences: list) -> tuple[np.ndarray, np.ndarray]:
        """Given a sequence of (repeated) unit ids, return the unique-unit array
        (``list(set(...))`` order) and the per-unit occurrence count aligned to it."""
        units = np.array(list(set(int(u) for u in unit_occurrences)))
        counts: dict[int, int] = {}
        for u in unit_occurrences:
            counts[int(u)] = counts.get(int(u), 0) + 1
        activity = np.array([counts.get(int(u), 0) for u in units])
        return units, activity

    def _fold_index_splits(
        self, units: np.ndarray, activity: np.ndarray | None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The single KFold / StratifiedKFold call shared by ``split_interactions``
        and ``split_participant_ids`` so the fold boundaries are defined once."""
        if self.stratified and activity is not None:
            activity_labels = np.digitize(activity, np.percentile(activity, [33, 67]))
            return list(self.splitter.split(units, activity_labels))
        return list(self.splitter.split(units))

    def split_participant_ids(self, df) -> list[set]:
        """K sets of *training* ``Participant ID`` values, one per fold.

        Mirrors ``split_interactions``' user split exactly (same first-appearance
        ``user_to_idx``, same RATING-row activity counts, same
        ``_fold_index_splits``) but expressed as participant IDs, for the
        per-fold causal-discovery precompute which works from the raw dataframe.
        """
        unique_users = list(df["Participant ID"].unique())  # dataloader's order
        user_to_idx = {u: i for i, u in enumerate(unique_users)}
        idx_to_user = {i: u for u, i in user_to_idx.items()}

        rated = df.loc[df["Rating score"].notna(), "Participant ID"]
        occurrences = [user_to_idx[p] for p in rated]
        units, activity = self._units_and_activity(occurrences)

        folds = []
        for train_idx, _ in self._fold_index_splits(units, activity):
            folds.append({idx_to_user[int(units[i])] for i in train_idx})
        return folds

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
        users, activity_values = self._units_and_activity(
            [interaction[0] for interaction in primary_data]
        )

        splits = self._fold_index_splits(
            users, activity_values if self.stratified else None
        )

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

        # Per-fold causal discovery: on for --causal / --ace CV runs unless the
        # --cv_global_causal escape hatch is set. When on, each fold's SEM head
        # is built from a DAG fitted on that fold's training participants only.
        uses_causal = getattr(args, "causal", False) or getattr(args, "ace", False)
        self._per_fold_causal = uses_causal and not getattr(args, "cv_global_causal", False)
        self._causal_config_path = getattr(args, "causal_config", None) or CAUSAL_CONFIG_PATH
        self._causal_fold_key = (
            causal_fold_key(n_folds, stratified, self._causal_config_path)
            if self._per_fold_causal
            else None
        )
        self._fold_causal_paths: list = []
        self._fold_norm_lookups: list = []
        self._global_norm_lookup: dict = {}

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
            "per_fold_causal": self._per_fold_causal,
            "causal_fold_key": self._causal_fold_key,
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

        if "idx_to_user" not in metadata and "user_to_idx" in metadata:
            metadata["idx_to_user"] = {i: u for u, i in metadata["user_to_idx"].items()}
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
            "user_to_idx": dataset.user_to_idx,
            "idx_to_user": dataset.idx_to_user,
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
            if len(interaction_list) == 0:
                adj_matrices[interaction_type] = []
                continue

            arr = np.asarray(interaction_list)  # [n_rows, 2 + n_components]
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

        fold_args = self.args
        if self._per_fold_causal:
            # Re-express the globally z-scored RATING/VQ/SF targets in this
            # fold's z-space so they match the fold's SEM head output space.
            f_lookup = self._fold_norm_lookups[fold_idx]
            train_data = refold_normalize_interactions(
                train_data, self._global_norm_lookup, f_lookup
            )
            val_data = refold_normalize_interactions(
                val_data, self._global_norm_lookup, f_lookup
            )

            # Copy this fold's causal artifact set into the checkpoint so the
            # fold dir is self-contained, and point the trainer at that copy.
            fold_causal_dir = fold_dir / "causal"
            fold_causal_dir.mkdir(exist_ok=True)
            src_models = self._fold_causal_paths[fold_idx].parent
            for name in (ROBUST_NPY, NORM_STATS_CSV):
                src = src_models / name
                if src.exists():
                    shutil.copy2(src, fold_causal_dir / name)
            fold_args = copy.copy(self.args)
            fold_args.causal_path = str(fold_causal_dir / ROBUST_NPY)
            print(f"  [per-fold causal] fold {fold_idx + 1} SEM head <- {fold_args.causal_path}")

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
            fold_args, fold_train_data, fold_val_data, metadata, fold_dir, fold_idx
        )

        # Run training
        fold_results = fold_trainer.train()

        return fold_results

    def _fold_train_participant_ids(self, fold_data: list, metadata: dict) -> list[set]:
        """Each fold's training ``Participant ID`` set, derived from the split
        just produced (RATING rows → user idx → ``idx_to_user``)."""
        idx_to_user = metadata.get("idx_to_user")
        if not idx_to_user:
            raise RuntimeError(
                "Per-fold causal discovery needs metadata['idx_to_user']; "
                "the dataset/cache did not provide a user_to_idx mapping."
            )
        out = []
        for train_interactions, _ in fold_data:
            rating_rows = train_interactions.get("RATING")
            if not rating_rows:
                raise RuntimeError(
                    "Per-fold causal discovery requires the RATING interaction "
                    "in the fold split; none found."
                )
            out.append({idx_to_user[int(r[0])] for r in rating_rows})
        return out

    def _prepare_per_fold_causal(self, fold_data: list, metadata: dict) -> None:
        """Resolve one causal artifact per fold (build any that are missing) and
        load the fold + global normalization lookups used by ``run_fold``."""
        key = self._causal_fold_key
        train_ids_by_fold = self._fold_train_participant_ids(fold_data, metadata)

        manifest = read_folds_manifest(key)
        if manifest is not None:
            assert_split_matches(manifest, train_ids_by_fold)
        else:
            strat_flag = "--stratified" if self.stratified else "--no-stratified"
            print(
                f"[per-fold causal] no precomputed manifest for key {key}; "
                f"building folds on demand. To precompute up front:\n"
                f"    python src/run_causal_discovery.py --cv --n_folds {self.n_folds} "
                f"{strat_flag} --causal_config {self._causal_config_path}"
            )

        n_jobs = getattr(self.args, "cv_causal_jobs", CV_CAUSAL_JOBS)
        self._fold_causal_paths = []
        self._fold_norm_lookups = []
        for i, train_ids in enumerate(train_ids_by_fold):
            npy = ensure_fold_artifact(
                i, train_ids, key, self._causal_config_path, n_jobs
            )
            self._fold_causal_paths.append(npy)
            self._fold_norm_lookups.append(norm_lookup_from_csv(npy.parent / NORM_STATS_CSV))

        if manifest is None:
            records = [
                {
                    "idx": i,
                    "n_train_participants": len(ids),
                    "train_participant_ids": sorted(str(p) for p in ids),
                    "artifact_path": str(self._fold_causal_paths[i]),
                    "causal_order": None,
                    "n_edges": None,
                }
                for i, ids in enumerate(train_ids_by_fold)
            ]
            write_folds_manifest(
                key, self.n_folds, self.stratified, records, self._causal_config_path
            )

        # Global interaction-target stats (for the fold-local renorm inversion).
        g_df = load_interaction_norm_stats(
            getattr(self.args, "causal_path", CAUSAL_MODEL_PATH)
        )
        self._global_norm_lookup = (
            {
                str(r["feature"]): (float(r["mean"]), float(r["std"]))
                for _, r in g_df.iterrows()
            }
            if g_df is not None
            else {}
        )
        if not self._global_norm_lookup:
            print(
                "[per-fold causal] WARNING: no global normalization stats; "
                "interaction targets stay on their as-loaded normalization."
            )

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

        # Per-fold causal discovery: resolve (or build) one DAG per fold, fitted
        # on that fold's training participants, before any training starts.
        if self._per_fold_causal:
            self._prepare_per_fold_causal(fold_data, metadata)

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
        """Save cross-validation results, summary CSV, markdown report, and metric plots."""
        with open(self.cv_dir / "cv_results.json", "w") as f:
            json.dump(self.cv_results, f, indent=2, default=str)

        # Save summary CSV
        self._save_cv_summary()

        # Save report and plots
        create_cv_report(self.cv_results, self.cv_dir)
        plot_cv_metrics(self.cv_results, self.cv_dir)

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
        print_cv_summary(self.cv_results)


def load_cv_results(cv_dir: str | Path) -> dict:
    """Load cross-validation results from a directory or JSON file."""
    cv_path = Path(cv_dir)
    results_file = cv_path if cv_path.is_file() else cv_path / "cv_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"CV results not found at {results_file}")

    with open(results_file, "r") as f:
        results = json.load(f)

    return results


def plot_cv_metrics(results: dict, save_dir: Path | str) -> None:
    """Plot cross-validation metrics across folds."""
    try:
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        fold_results = results.get("fold_results", [])
        if not fold_results:
            return

        n_folds = len(fold_results)
        metrics = list(fold_results[0].get("best_metrics", {}).keys())
        if not metrics:
            return

        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for i, metric in enumerate(metrics):
            values = [fold.get("best_metrics", {}).get(metric, 0.0) for fold in fold_results]
            mean_val = float(np.mean(values))

            ax = axes[i]
            ax.bar(range(1, n_folds + 1), values, alpha=0.7)
            ax.axhline(y=mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.4f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} across folds")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(save_dir / "cv_metrics.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("CV metrics plot saved.")

    except ImportError:
        print("matplotlib not available, skipping plots")


def create_cv_report(results: dict, save_dir: Path | str) -> None:
    """Create comprehensive CV report in markdown format."""
    save_dir = Path(save_dir)
    report = []

    fold_results = results.get("fold_results", [])
    n_folds = results.get("n_folds", len(fold_results))
    cv_time = results.get("cv_time", 0.0)
    best_fold = results.get("best_fold", 0) + 1
    worst_fold = results.get("worst_fold", 0) + 1

    report.append("# Cross-Validation Report\n\n")
    report.append(f"**Number of folds:** {n_folds}\n")
    report.append(f"**Total CV time:** {cv_time:.2f}s\n")
    report.append(f"**Best fold:** {best_fold}\n")
    report.append(f"**Worst fold:** {worst_fold}\n\n")

    report.append("## Mean ± Std Results\n\n")
    report.append("| Metric | Mean | Std |\n")
    report.append("|--------|------|-----|\n")

    mean_metrics = results.get("mean_metrics", {})
    std_metrics = results.get("std_metrics", {})
    for metric, mean_val in mean_metrics.items():
        std_val = std_metrics.get(metric, 0.0)
        report.append(f"| {metric} | {mean_val:.4f} | {std_val:.4f} |\n")

    report.append("\n## Per-Fold Results\n\n")
    metric_keys = list(mean_metrics.keys())
    if metric_keys:
        report.append("| Fold | " + " | ".join(metric_keys) + " | Training Time |\n")
        report.append("|------|" + "|".join(["-" * 10] * (len(metric_keys) + 1)) + "|\n")

        for i, fold_result in enumerate(fold_results):
            row = f"| {i+1} |"
            for metric in metric_keys:
                val = fold_result.get("best_metrics", {}).get(metric, 0.0)
                row += f" {val:.4f} |"
            tt = fold_result.get("training_time", 0.0)
            row += f" {tt:.2f}s |\n"
            report.append(row)

    with open(save_dir / "cv_report.md", "w") as f:
        f.writelines(report)

    print("CV report saved.")


def print_cv_summary(results: dict) -> None:
    """Print summary of CV results."""
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")

    fold_results = results.get("fold_results", [])
    n_folds = results.get("n_folds", len(fold_results))
    cv_time = results.get("cv_time", 0.0)
    best_fold = results.get("best_fold", 0) + 1
    worst_fold = results.get("worst_fold", 0) + 1

    print(f"Number of folds: {n_folds}")
    print(f"Total CV time: {cv_time:.2f}s")
    print(f"Best fold: {best_fold}")
    print(f"Worst fold: {worst_fold}")

    print("\nMean ± Std across folds:")
    print("-" * 40)
    for metric_name, mean_val in results.get("mean_metrics", {}).items():
        std_val = results.get("std_metrics", {}).get(metric_name, 0.0)
        print(f"{metric_name:15s}: {mean_val:.4f} ± {std_val:.4f}")

    print("\nPer-fold results:")
    print("-" * 40)
    for i, fold_result in enumerate(fold_results):
        print(f"Fold {i+1}:")
        for metric_name, value in fold_result.get("best_metrics", {}).items():
            print(f"  {metric_name:13s}: {value:.4f}")


def analyze_cv_results(cv_dir: str | Path) -> dict:
    """Load and analyze CV results: generate plots, markdown report, and print summary."""
    cv_path = Path(cv_dir)
    target_dir = cv_path if cv_path.is_dir() else cv_path.parent

    results = load_cv_results(cv_path)

    print(f"\nAnalyzing cross-validation results in {target_dir}...")
    print("Creating plots...")
    plot_cv_metrics(results, target_dir)

    print("Creating report...")
    create_cv_report(results, target_dir)

    print_cv_summary(results)
    print(f"\nAnalysis saved to: {target_dir}")
    return results


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
    """Main cross-validation and analysis entry point.

    Supports:
      1. Training / CV: delegates to utility.parser (e.g. `python cross_validation.py --cv`)
      2. Post-hoc analysis: `python cross_validation.py --analyze <cv_dir>`
         or `python cross_validation.py <cv_dir>`
    """
    import sys

    # Support standalone analysis flags: python cross_validation.py --analyze <cv_dir>
    if len(sys.argv) > 1 and sys.argv[1] in ("--analyze", "-a", "--cv_dir"):
        if len(sys.argv) > 2:
            analyze_cv_results(sys.argv[2])
        else:
            print("Error: Please provide a CV directory path to analyze.")
            sys.exit(1)
        return

    # Check if a single argument is passed pointing to an existing CV directory
    if len(sys.argv) == 2 and (Path(sys.argv[1]) / "cv_results.json").exists():
        analyze_cv_results(sys.argv[1])
        return

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
