"""
Evaluation script for trained GHCF model.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from configs.config import (
    ACE_FIXED_WEIGHTS,
    ACE_WEIGHT_MODE,
    CAUSAL_CONFIG_PATH,
    CAUSAL_MODEL_PATH,
    DEVICE,
    MODEL_CONFIG,
    RANDOM_SEED,
)
from models.GHCF import GHCF
from models.GHCF_ACE import GHCF_ACE
from models.GHCF_CE import GHCF_CE
from utility.batch_processing import BatchProcessor
from utility.dataloader import load_processed_data
from utility.helper import resolve_causal_paths, set_seed


class Evaluator:
    """Evaluator class for trained GHCF models."""

    def __init__(self, args):
        """Initialize evaluator with command line arguments."""
        self.args = args
        self.device = DEVICE
        self.k_values = ast.literal_eval(args.k_values)
        self.model: Optional[GHCF] = None
        self.train_data: Optional[Dict] = None
        self.test_data: Optional[Dict] = None
        self.metadata: Optional[Dict] = None

        # Set default values for causal and ace if not provided
        if not hasattr(self.args, "causal"):
            self.args.causal = False
        if not hasattr(self.args, "ace"):
            self.args.ace = False
        if not hasattr(self.args, "causal_path"):
            self.args.causal_path = CAUSAL_MODEL_PATH
        if not hasattr(self.args, "causal_config"):
            self.args.causal_config = CAUSAL_CONFIG_PATH

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def setup(self) -> None:
        """Setup evaluator by loading model config, data, and weights."""
        set_seed(RANDOM_SEED)
        self.logger.info(f"Using device: {self.device}")

        model_config = self._load_model_config()
        self._load_data(model_config["interaction_types"])
        self._load_model(model_config)

    def _load_model_config(self) -> Dict:
        """Load model configuration from results.json."""
        model_dir = Path(self.args.model_dir)
        results_file = model_dir / "results.json"

        if not results_file.exists():
            raise FileNotFoundError(f"Model results file not found: {results_file}")

        with open(results_file, "r") as f:
            results = json.load(f)

        if "model_config" not in results:
            raise ValueError("Cannot find model_config in results.json")

        return results["model_config"]

    def _get_processed_data_dir(self, interaction_types: List[str]) -> Path:
        """Determine the correct processed data directory based on interaction types."""
        data_base_path = Path("data")

        # Try different directory naming patterns
        patterns = [
            (
                f"processed_{'_'.join(sorted(interaction_types))}"
                if len(interaction_types) > 1
                else None
            ),
            f"processed_{'_'.join(interaction_types)}" if len(interaction_types) > 1 else None,
            f"processed_{interaction_types[0]}" if len(interaction_types) == 1 else None,
            "processed",
        ]

        for pattern in filter(None, patterns):
            processed_dir = data_base_path / pattern
            if processed_dir.exists():
                return processed_dir

        # List available directories for debugging
        available_dirs = [
            d.name
            for d in data_base_path.iterdir()
            if d.is_dir() and d.name.startswith("processed")
        ]
        raise FileNotFoundError(
            f"No matching processed data directory found for interactions {interaction_types}. "
            f"Available directories: {available_dirs}"
        )

    def _load_data(self, interaction_types: List[str]) -> None:
        """Load processed data."""
        processed_data_dir = self._get_processed_data_dir(interaction_types)
        self.logger.info(f"Using processed data directory: {processed_data_dir}")

        self.logger.info("Loading processed data...")
        self.train_data, self.test_data, self.metadata = load_processed_data(processed_data_dir)

        self.logger.info(
            f"Data loaded: {self.metadata['n_users']} users, "
            f"{self.metadata['n_items']} items, "
            f"Interaction types: {self.metadata['interaction_types']}"
        )

    def _load_model(self, model_config: Dict) -> None:
        """Load trained model from checkpoint."""
        model_dir = Path(self.args.model_dir)

        # Create model based on arguments - similar to train.py
        use_ace = getattr(self.args, "ace", False)
        common_kwargs = dict(
            n_users=model_config["n_users"],
            n_items=model_config["n_items"],
            n_user_features=MODEL_CONFIG["user_feature_dim"],
            n_item_features=MODEL_CONFIG["item_feature_dim"],
            interaction_types=model_config["interaction_types"],
            embedding_dim=model_config["graph_config"]["embedding_dim"],
            layer_sizes=model_config["graph_config"]["layer_sizes"],
            dropout_rates=model_config["graph_config"]["dropout_rates"],
            gnn_type=model_config["graph_config"]["gnn_type"],
            n_heads=model_config["graph_config"].get("n_heads", 4),
            device=self.device,
        )

        if not self.args.causal and not use_ace:
            self.logger.info("Using standard GHCF model")
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
            )

            if use_ace:
                self.logger.info("Using GHCF_ACE model")
                self.model = GHCF_ACE(
                    **causal_kwargs,
                    ace_weight_mode=getattr(self.args, "ace_weight_mode", ACE_WEIGHT_MODE),
                    ace_fixed_weights=ACE_FIXED_WEIGHTS,
                ).to(self.device)
            else:
                self.logger.info("Using GHCF_CE model")
                self.model = GHCF_CE(**causal_kwargs).to(self.device)
        # Load checkpoint
        checkpoint_path = model_dir / self.args.checkpoint
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Loaded model from {checkpoint_path}")
            self.logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
            self.logger.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def _setup_causal_paths(self) -> Tuple[str, str]:
        return resolve_causal_paths(self.args.causal_path, self.args.causal_config)

    def evaluate_recommendations(
        self,
        k_values: List[int] = [5, 10, 20],
        target_interaction: str = "RATING",
    ) -> Dict[str, float]:
        """
        Evaluate the model for the recommendation task.

        Delegates to BatchProcessor.evaluate_model, the same implementation
        used during training.

        Args:
            k_values: K values for ranking metrics
            target_interaction: Primary interaction for recommendation

        Returns:
            Dictionary of evaluation metrics
        """
        processor = BatchProcessor(self.metadata["n_users"], self.test_data["n_items"])
        self.logger.info("Evaluating test users...")

        return processor.evaluate_model(
            self.model,
            self.test_data["interactions"],
            self.train_data["interactions"],
            k_values=k_values,
            device=self.device,
            train_data=self.train_data,
            target_interaction=target_interaction,
            show_progress=True,
        )

    def save_results(self, metrics: Dict[str, float]) -> None:
        """
        Save evaluation results.

        Args:
            metrics: Evaluation metrics
        """
        model_dir = Path(self.args.model_dir)

        eval_results = {
            "checkpoint": self.args.checkpoint,
            "k_values": self.k_values,
            "metrics": metrics,
        }

        eval_file = model_dir / f'eval_results_{self.args.checkpoint.split(".")[0]}.json'
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        self.logger.info(f"Evaluation results saved to {eval_file}")

    def print_results(self, metrics: Dict[str, float]) -> None:
        """Print evaluation results."""
        self.logger.info("Evaluation Results:")
        self.logger.info("-" * 50)
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"{metric_name:15s}: {metric_value:.4f}")

    def run(self) -> None:
        """Run evaluation process."""
        # Setup components
        self.setup()

        # Evaluate model
        self.logger.info(f"Evaluating model with K values: {self.k_values}")

        metrics = self.evaluate_recommendations(self.k_values)

        # Print and save results
        self.print_results(metrics)
        self.save_results(metrics)


def main():
    """Main evaluation function (legacy support)."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GHCF model")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory containing trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_state.pth",
        choices=["best_state.pth", "last_state.pth"],
        help="Checkpoint to load",
    )
    parser.add_argument("--k_values", type=str, default="[3,5,10]", help="K values for evaluation")
    parser.add_argument(
        "--interactions",
        type=str,
        nargs="+",
        default=["RATING", "BFI_PERCEPTION", "VOICE_QUALITY", "SUBJECTIVE_FEELING"],
        help="Interaction types to evaluate",
    )
    # Add causal and ACE arguments similar to train.py
    parser.add_argument("--causal", action="store_true", help="Use GHCF_CE model")
    parser.add_argument("--ace", action="store_true", help="Use GHCF_ACE model")
    parser.add_argument(
        "--ace_weight_mode",
        type=str,
        default=ACE_WEIGHT_MODE,
        choices=["learnable", "fixed"],
        help="GHCF-ACE GNN/causal mixing mode (must match training)",
    )
    parser.add_argument(
        "--causal_path",
        type=str,
        default=CAUSAL_MODEL_PATH,
        help="Path to causal model file",
    )
    parser.add_argument(
        "--causal_config",
        type=str,
        default=CAUSAL_CONFIG_PATH,
        help="Path to causal config file",
    )

    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
