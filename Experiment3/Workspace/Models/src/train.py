"""
Training script for GHCF model with Trainer class.
"""

import ast
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from configs.config import (
    ACE_FIXED_WEIGHTS,
    ACE_WEIGHT_MODE,
    BOUND_BFI_OUTPUT,
    BPR_LAMBDA,
    BPR_NUM_NEG,
    CAUSAL_MODEL_PATH,
    DATA_DIR,
    DATABASE_PATH,
    DEVICE,
    EVAL_INTERVAL,
    FREEZE_SEM,
    GRAPH_CONFIG,
    LOSS_MODE,
    NORMALIZE_INTERACTIONS,
    RANDOM_SEED,
    RATING_THRESHOLD,
    TEST_SIZE,
)
from models.GHCF import GHCF
from models.GHCF_ACE import GHCF_ACE
from models.GHCF_CE import GHCF_CE
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utility.batch_processing import BatchProcessor
from utility.dataloader import eHMIDataset, load_processed_data
from utility.gradnorm import GradNorm
from utility.helper import load_interaction_norm_stats, resolve_causal_paths, set_seed


class EarlyStoppingHandler:
    """Early stopping handler for training."""

    def __init__(self, patience: int = 100, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if early stopping should be triggered."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    """Trainer class for GHCF models."""

    def __init__(self, args):
        """
        Initialize trainer.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = DEVICE
        self.k_values = ast.literal_eval(args.k_values)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.batch_processor = None
        self.early_stopping = None
        self.gradnorm_handler = None
        self.writer = None
        self.model_dir = None

        # Data
        self.train_data = None
        self.test_data = None
        self.n_users = 0
        self.n_items = 0
        self.interaction_types = []

        # Training state
        self.current_interaction_weights = {}
        self.best_metrics = {}
        self.best_score = float("-inf")
        self.loss_history = {"total": []}
        self.weight_history = {}

    def setup(self) -> None:
        """Setup trainer components."""
        # Set random seed
        set_seed(RANDOM_SEED)
        print(f"Using device: {self.device}")

        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = Path(f"checkpoints/model_{timestamp}")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.writer = SummaryWriter(log_dir=self.model_dir / "logs")

        # Load or create dataset
        self._setup_data()

        # Create model
        self._create_model()

        # Setup training components
        self._setup_training_components()

    def _setup_data(self) -> None:
        """Load or create dataset."""
        data_path = DATABASE_PATH
        interactions_hash = "_".join(sorted(self.args.interactions))
        # Scope the cache to the active data subset (SET) so 25/35 never collide.
        processed_data_dir = DATA_DIR / f"processed_{interactions_hash}"

        if processed_data_dir.exists() and (processed_data_dir / "metadata.pkl").exists():
            print(f"Loading processed data for interactions: {self.args.interactions}")
            self.train_data, self.test_data, metadata = load_processed_data(processed_data_dir)
            self.n_users = metadata["n_users"]
            self.n_items = metadata["n_items"]
            self.interaction_types = metadata["interaction_types"]

            # Verify cached data matches requested interactions
            if set(self.interaction_types) != set(self.args.interactions):
                print("Cached interaction types don't match. Creating new dataset...")
                self._create_new_dataset(data_path, processed_data_dir)
        else:
            print(f"Creating new dataset for interactions: {self.args.interactions}")
            self._create_new_dataset(data_path, processed_data_dir)

        print("Dataset loaded:")
        print(f"  - Users: {self.n_users}")
        print(f"  - Items: {self.n_items}")
        print(f"  - Interaction types: {self.interaction_types}")

    def _create_new_dataset(self, data_path: Path, processed_data_dir: Path) -> None:
        """Create new dataset."""
        # Interaction-target normalization is independent of feature
        # normalization; share the causal model's stats so RATING/VQ/SF targets
        # match the SEM head's output space.
        normalize_interactions = getattr(
            self.args, "normalize_interactions", NORMALIZE_INTERACTIONS
        )
        norm_stats = load_interaction_norm_stats(
            getattr(self.args, "causal_path", CAUSAL_MODEL_PATH)
        )
        dataset = eHMIDataset(
            data_path=str(data_path),
            interaction_types=self.args.interactions,
            test_size=TEST_SIZE,
            adj_type=self.args.adj_type,
            random_state=RANDOM_SEED,
            normalize_interactions=normalize_interactions,
            interaction_norm_stats=norm_stats,
        )

        self.train_data = dataset.get_train_data()
        self.test_data = dataset.get_test_data()
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.interaction_types = dataset.interaction_types

        # Save processed data
        dataset.save_processed_data(processed_data_dir)

    def _create_model(self) -> None:
        """Create model based on arguments."""
        use_ace = getattr(self.args, "ace", False)
        common_kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_user_features=self.train_data["user_features"].shape[1],
            n_item_features=self.train_data["item_features"].shape[1],
            interaction_types=self.interaction_types,
            embedding_dim=self.args.embed_size,
            layer_sizes=GRAPH_CONFIG["layer_sizes"],
            dropout_rates=GRAPH_CONFIG["dropout_rates"],
            gnn_type=getattr(self.args, "gnn_type", GRAPH_CONFIG["gnn_type"]),
            n_heads=getattr(self.args, "n_heads", GRAPH_CONFIG.get("n_heads", 4)),
            device=self.device,
            bound_bfi_output=getattr(self.args, "bound_bfi", BOUND_BFI_OUTPUT),
        )

        if not self.args.causal and not use_ace:
            self.model = GHCF(**common_kwargs).to(self.device)

        else:
            if self.args.causal and use_ace:
                raise ValueError("Cannot use both --causal and --ace flags simultaneously")

            causal_model_path, causal_config_path = self._setup_causal_paths()
            causal_kwargs = dict(
                common_kwargs,
                use_causal=True,
                causal_model_path=causal_model_path,
                causal_config_path=causal_config_path,
                trainable_sem=not getattr(self.args, "freeze_sem", FREEZE_SEM),
            )

            if use_ace:
                self.model = GHCF_ACE(
                    **causal_kwargs,
                    ace_weight_mode=getattr(self.args, "ace_weight_mode", ACE_WEIGHT_MODE),
                    ace_fixed_weights=self._resolve_ace_fixed_weights(),
                ).to(self.device)
            else:
                self.model = GHCF_CE(**causal_kwargs).to(self.device)

        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _resolve_ace_fixed_weights(self) -> Dict[str, Dict[str, float]]:
        """Per-type fixed ACE weights: config defaults, optionally overridden by
        global --ace_w_mlp / --ace_w_causal applied to every non-BFI type."""
        weights = {k: dict(v) for k, v in ACE_FIXED_WEIGHTS.items()}
        w_mlp = getattr(self.args, "ace_w_mlp", None)
        w_causal = getattr(self.args, "ace_w_causal", None)
        if w_mlp is not None or w_causal is not None:
            for t in weights:
                if w_mlp is not None:
                    weights[t]["w_mlp"] = w_mlp
                if w_causal is not None:
                    weights[t]["w_causal"] = w_causal
        return weights

    def _setup_causal_paths(self) -> Tuple[str, str]:
        return resolve_causal_paths(self.args.causal_path, self.args.causal_config)

    def _configure_rating_loss(self) -> None:
        """Configure the RATING loss mode (mse / hybrid) on the model.

        In hybrid mode the model also needs each user's positively-rated items
        (rating >= RATING_THRESHOLD) to sample BPR negatives.
        """
        loss_mode = getattr(self.args, "loss", LOSS_MODE)
        self.model.loss_mode = loss_mode
        self.model.bpr_lambda = getattr(self.args, "bpr_lambda", BPR_LAMBDA)
        self.model.bpr_num_neg = BPR_NUM_NEG
        self.model.rating_threshold = RATING_THRESHOLD

        if loss_mode == "hybrid":
            pos_items: Dict[int, set] = {}
            for row in self.train_data["interactions"].get("RATING", []):
                if len(row) >= 3 and float(row[2]) >= RATING_THRESHOLD:
                    pos_items.setdefault(int(row[0]), set()).add(int(row[1]))
            self.model.rating_pos_items = pos_items
            print(
                f"RATING loss = hybrid (MSE + {self.model.bpr_lambda} * BPR); "
                f"positives for {len(pos_items)} users"
            )
        else:
            print("RATING loss = mse")

    def _setup_training_components(self) -> None:
        """Setup training components."""
        # Setup batch processor
        self.batch_processor = BatchProcessor(self.n_users, self.n_items, self.args.batch_size)

        # Configure RATING loss mode (mse / hybrid BPR) on the model
        self._configure_rating_loss()

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.decay
        )

        # Setup learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epoch,
            eta_min=self.args.lr * 0.01,  # Minimum LR is 1% of initial LR
        )

        # Setup early stopping: patience is given in epochs, but improvement is
        # only observable when the model is evaluated (every EVAL_INTERVAL epochs)
        self.early_stopping = EarlyStoppingHandler(max(1, self.args.patience // EVAL_INTERVAL))

        # Setup GradNorm if enabled
        use_gradnorm = getattr(self.args, "use_gradnorm", False) or GRAPH_CONFIG.get(
            "use_gradnorm", False
        )
        self.current_interaction_weights = GRAPH_CONFIG.get("interaction_weights", {})

        if use_gradnorm:
            print("Initializing GradNorm for dynamic loss balancing...")
            gradnorm_alpha = getattr(
                self.args, "gradnorm_alpha", GRAPH_CONFIG.get("gradnorm_alpha", 1.5)
            )
            self.gradnorm_handler = GradNorm(
                task_names=self.interaction_types,
                alpha=gradnorm_alpha,
                initial_task_weights=self.current_interaction_weights,
                device=self.device,
            )
            print(f"GradNorm initialized with alpha={gradnorm_alpha}")
        else:
            print("Using static interaction weights:", self.current_interaction_weights)

        # Initialize loss and weight histories
        for interaction_type in self.interaction_types:
            self.loss_history[interaction_type] = []
            self.weight_history[interaction_type] = []

        # In hybrid mode, also track the RATING MSE/BPR components so they are
        # plotted alongside the combined RATING loss.
        if getattr(self.model, "loss_mode", "mse") == "hybrid" and "RATING" in self.interaction_types:
            self.loss_history.setdefault("RATING_mse", [])
            self.loss_history.setdefault("RATING_bpr", [])

    def save_checkpoint(
        self, epoch: int, loss: float, metrics: dict, is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "metrics": metrics,
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.model_dir / "last_state.pth")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.model_dir / "best_state.pth")

    def load_checkpoint(self) -> int:
        """Load checkpoint if exists."""
        if self.args.pretrain and (self.model_dir / "last_state.pth").exists():
            print("Loading checkpoint...")
            checkpoint_path = self.model_dir / "last_state.pth"
            # Fix for PyTorch 2.6+ weights_only default change
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state if available
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            self.best_metrics = checkpoint.get("metrics", {})
            self.best_score = self.best_metrics.get("recall@10", float("-inf"))

            return start_epoch

        return 0

    def train_epoch(self, epoch_num: int) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_individual_losses = {}
        n_batches = 0

        # Create training batches
        batches = self.batch_processor.create_training_batches(self.train_data["interactions"])
        pbar = tqdm(batches, desc=f"Epoch {epoch_num}")

        for batch_data in pbar:
            self.optimizer.zero_grad()

            # Move batch to device
            for key in ["users", "items"]:
                batch_data[key] = batch_data[key].to(self.device)
            batch_data["values"] = [
                torch.FloatTensor(v).to(self.device) for v in batch_data["values"]
            ]

            # Forward pass
            user_embeddings, item_embeddings, relation_embeddings = self.model(
                self.train_data["user_features"].to(self.device),
                self.train_data["item_features"].to(self.device),
                {
                    k: [matrix.to(self.device) for matrix in matrices]
                    for k, matrices in self.train_data["adj_matrices"].items()
                },
            )

            # Compute loss
            use_gradnorm = self.gradnorm_handler is not None

            loss, individual_losses = self.model.compute_loss(
                user_embeddings,
                item_embeddings,
                relation_embeddings,
                batch_data,
                self.current_interaction_weights,
                return_individual_losses=True,
                user_features=self.train_data["user_features"].to(
                    self.device
                ),  # Add user_features for causal models
                use_gradnorm=use_gradnorm,
                gradnorm_handler=self.gradnorm_handler,
            )

            # Update weights if using GradNorm
            if use_gradnorm:
                self.current_interaction_weights = self.gradnorm_handler.get_current_weights()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            n_batches += 1

            for interaction_type, loss_val in individual_losses.items():
                if interaction_type not in total_individual_losses:
                    total_individual_losses[interaction_type] = 0.0
                total_individual_losses[interaction_type] += loss_val

            # Diagnostic RATING loss components (hybrid mode): accumulated for
            # logging only -- they are not part of the optimized/GradNorm dict.
            for comp_name, comp_val in (
                getattr(self.model, "last_rating_components", {}) or {}
            ).items():
                total_individual_losses[comp_name] = (
                    total_individual_losses.get(comp_name, 0.0) + comp_val
                )

            # Update progress bar
            self._update_progress_bar(pbar, loss.item(), individual_losses)

        # Calculate averages
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_individual_losses = {k: v / n_batches for k, v in total_individual_losses.items()}

        return avg_loss, avg_individual_losses, self.current_interaction_weights

    def _update_progress_bar(self, pbar, current_loss: float, individual_losses: dict) -> None:
        """Update progress bar with loss information."""
        loss_str = f"Total: {current_loss:.4f}"
        if individual_losses:
            individual_losses = dict(sorted(individual_losses.items()))
            loss_details = ", ".join([f"{k}: {v:.3f}" for k, v in individual_losses.items()])
            loss_str += f" | {loss_details}"
        pbar.set_postfix_str(loss_str)

    def evaluate_model(self, k_values: List[int]) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        metrics = self.batch_processor.evaluate_model(
            self.model,
            self.test_data["interactions"],
            self.train_data["interactions"],
            k_values,
            self.device,
            self.train_data,
        )
        return metrics

    def log_metrics(
        self, epoch: int, train_loss: float, individual_losses: dict, eval_metrics: dict = None
    ) -> None:
        """Log metrics to TensorBoard."""
        # Log training losses
        self.writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        for interaction_type, loss_val in individual_losses.items():
            self.writer.add_scalar(f"Loss/Train_{interaction_type}", loss_val, epoch)

        # Log learning rate
        current_lr = self.scheduler.get_last_lr()[0]
        self.writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Log weights
        for interaction_type, weight in self.current_interaction_weights.items():
            self.writer.add_scalar(f"Weights/{interaction_type}", weight, epoch)

        # Log GHCF-ACE GNN/causal mixing weights (per interaction type)
        if hasattr(self.model, "get_ace_weights"):
            for itype, (w_mlp, w_causal) in self.model.get_ace_weights().items():
                self.writer.add_scalar(f"ACE/{itype}_w_mlp", w_mlp, epoch)
                self.writer.add_scalar(f"ACE/{itype}_w_causal", w_causal, epoch)

        # Log evaluation metrics
        if eval_metrics:
            for metric_name, metric_value in eval_metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

    def update_histories(self, train_loss: float, individual_losses: dict) -> None:
        """Update loss and weight histories."""
        self.loss_history["total"].append(train_loss)
        for interaction_type, loss_val in individual_losses.items():
            if interaction_type in self.loss_history:
                self.loss_history[interaction_type].append(loss_val)

        for interaction_type in self.interaction_types:
            weight = self.current_interaction_weights.get(interaction_type, 1.0)
            self.weight_history[interaction_type].append(weight)

    def save_plots(self) -> None:
        """Save loss and weight plots."""
        try:
            from utility.plotting import save_loss_plots, save_weight_plots

            save_loss_plots(self.loss_history, self.model_dir)
            if self.gradnorm_handler:
                save_weight_plots(self.weight_history, self.model_dir)
            print("Training plots saved successfully")
        except ImportError as e:
            print(f"Warning: Could not import plotting utilities: {e}")
            print("Plots will not be saved, but training results are still available")
        except Exception as e:
            print(f"Warning: Error saving plots: {e}")
            print("Plots could not be saved, but training results are still available")

    def save_results(self, training_time: float, final_metrics: dict) -> None:
        """Save final training results."""
        results = {
            "args": vars(self.args),
            "model_config": {
                "n_users": self.n_users,
                "n_items": self.n_items,
                "interaction_types": self.interaction_types,
                "graph_config": GRAPH_CONFIG,
                "use_gradnorm": self.gradnorm_handler is not None,
                # Persist output-bounding so evaluate.py rebuilds an identical
                # forward path (the sigmoid changes the BFI head's outputs).
                "bound_bfi_output": getattr(self.model, "bound_bfi_output", BOUND_BFI_OUTPUT),
            },
            "best_metrics": self.best_metrics,
            "final_metrics": final_metrics,
            "training_time": training_time,
            "loss_history": self.loss_history,
            "weight_history": self.weight_history if self.gradnorm_handler else {},
            "final_weights": self.current_interaction_weights,
        }

        # Final GHCF-ACE GNN/causal mixing weights, if applicable
        if hasattr(self.model, "get_ace_weights"):
            results["ace_weight_mode"] = getattr(self.model, "ace_weight_mode", None)
            results["ace_weights"] = {
                itype: {"w_mlp": w_mlp, "w_causal": w_causal}
                for itype, (w_mlp, w_causal) in self.model.get_ace_weights().items()
            }

        with open(self.model_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    def print_progress(
        self,
        epoch: int,
        train_loss: float,
        individual_losses: dict,
        current_score: float,
        epoch_time: float,
    ) -> None:
        """Print training progress."""
        individual_loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in individual_losses.items()])

        weight_str = ""
        if self.gradnorm_handler and self.current_interaction_weights:
            weights_display = ", ".join(
                [f"{k}: {v:.3f}" for k, v in self.current_interaction_weights.items()]
            )
            weight_str = f" | Weights: {weights_display}"

        # GHCF-ACE: show the causal share per interaction type
        ace_str = ""
        if hasattr(self.model, "get_ace_weights"):
            ace_display = ", ".join(
                [f"{k}: {w_causal:.2f}" for k, (_, w_causal) in self.model.get_ace_weights().items()]
            )
            ace_str = f" | ACE w_causal: {ace_display}"

        # Get current learning rate
        current_lr = self.scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:4d} | Total Loss: {train_loss:.4f} | {individual_loss_str} | "
            f"Recall@10: {current_score:.4f} | LR: {current_lr:.6f}{weight_str}{ace_str} | "
            f"Time: {epoch_time:.2f}s"
        )

    def run(self) -> None:
        """Run training process."""
        # Setup components
        self.setup()

        # Load checkpoint if exists
        start_epoch = self.load_checkpoint()

        print("Starting training...")
        training_start_time = time.time()

        for epoch in range(start_epoch, self.args.epoch):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{self.args.epoch}")

            # Training
            train_loss, individual_losses, current_weights = self.train_epoch(epoch + 1)
            self.update_histories(train_loss, individual_losses)

            # Update learning rate scheduler
            self.scheduler.step()

            # Evaluation
            if epoch % EVAL_INTERVAL == 0:
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
                self.print_progress(
                    epoch, train_loss, individual_losses, current_score, epoch_time
                )

            else:
                # Just log training loss
                self.log_metrics(epoch, train_loss, individual_losses)

        # Final evaluation and cleanup
        self._finalize_training(training_start_time)

    def _finalize_training(self, training_start_time: float) -> None:
        """Finalize training process."""
        print("\nFinal evaluation on test set...")
        final_metrics = self.evaluate_model(self.k_values)

        # Save plots and results
        self.save_plots()

        training_time = time.time() - training_start_time
        self.save_results(training_time, final_metrics)

        # Print final results
        print(f"\nTraining completed in {training_time:.2f}s")
        print("Best metrics:")
        for metric_name, metric_value in self.best_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        print("Final metrics:")
        for metric_name, metric_value in final_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        # Close writer
        self.writer.close()


def main():
    """Main training function (legacy support)."""
    from utility.parser import parse_args

    args = parse_args()

    # Check if cross-validation is requested
    if hasattr(args, "cv") and args.cv:
        from cross_validation import CrossValidationTrainer

        cv_trainer = CrossValidationTrainer(args, n_folds=args.n_folds, stratified=args.stratified)
        cv_trainer.run()
    else:
        trainer = Trainer(args)
        trainer.run()


if __name__ == "__main__":
    main()
