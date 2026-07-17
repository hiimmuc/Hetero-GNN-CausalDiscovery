"""
Unified argument parser for the GHCF model.

Defaults are sourced from configs/config.py so the config file stays the
single source of truth; any command line argument overrides its config value.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to path so configs is importable when this module
# is loaded standalone
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    ACE_WEIGHT_MODE,
    ADJ_TYPE,
    BATCH_SIZE,
    BOUND_BFI_OUTPUT,
    CAUSAL_CONFIG_PATH,
    CAUSAL_MODEL_PATH,
    BPR_LAMBDA,
    CV_CONFIG,
    EARLY_STOP_PATIENCE,
    FREEZE_SEM,
    GRAPH_CONFIG,
    LEARNING_RATE,
    LOSS_MODE,
    NORMALIZE_INTERACTIONS,
    NUM_EPOCHS,
    WEIGHT_DECAY,
)


def create_base_parser() -> argparse.ArgumentParser:
    """Create base parser with common arguments."""
    parser = argparse.ArgumentParser(
        description="GHCF eHMI Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with defaults from configs/config.py
  python main.py train

  # Train with specific interactions
  python main.py train --interactions RATING BFI_PERCEPTION

  # Evaluate trained model
  python main.py evaluate --model_dir checkpoints/model_20241218_120000

  # Train with custom parameters (overrides config.py)
  python main.py train --embed_size 128 --lr 0.0001 --epoch 500
        """,
    )
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between train and evaluate."""
    parser.add_argument(
        "--interactions",
        type=str,
        nargs="+",
        default=["RATING", "BFI_PERCEPTION", "VOICE_QUALITY", "SUBJECTIVE_FEELING"],
        help="Interaction types to use",
    )
def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training-related arguments (defaults from configs/config.py)."""
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epoch", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="Early stopping patience in epochs without improvement",
    )
    parser.add_argument("--decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument(
        "--loss",
        type=str,
        default=LOSS_MODE,
        choices=["mse", "hybrid"],
        help="RATING loss mode: 'mse' (magnitude only) or 'hybrid' (MSE + BPR ranking). "
        "BFI/VQ/SF always use MSE.",
    )
    parser.add_argument(
        "--bpr_lambda",
        type=float,
        default=BPR_LAMBDA,
        help="Weight of the BPR ranking term when --loss hybrid",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=GRAPH_CONFIG["embedding_dim"],
        help="Embedding dimension size",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default=GRAPH_CONFIG["gnn_type"],
        choices=["gcn", "sage", "gat"],
        help="GNN layer type for message passing",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=GRAPH_CONFIG.get("n_heads", 4),
        help="Number of attention heads (only used when --gnn_type gat)",
    )
    parser.add_argument("--pretrain", type=int, default=0, help="Use pretrained model")
    parser.add_argument(
        "--bound_bfi",
        action=argparse.BooleanOptionalAction,
        default=BOUND_BFI_OUTPUT,
        help=(
            "Squash the BFI perception head to [0, 1] with a sigmoid so it matches "
            "the MinMax BFI targets and the causal head's input domain. Use "
            "--no-bound_bfi to leave it unbounded (the original linear head)."
        ),
    )


def add_gradnorm_arguments(parser: argparse.ArgumentParser) -> None:
    """Add GradNorm-related arguments."""
    parser.add_argument(
        "--use_gradnorm", action="store_true", help="Use GradNorm for dynamic loss balancing"
    )
    parser.add_argument(
        "--gradnorm_alpha",
        type=float,
        default=GRAPH_CONFIG["gradnorm_alpha"],
        help="GradNorm restoring force hyperparameter (typically 0.12-3.0)",
    )


def add_causal_arguments(parser: argparse.ArgumentParser) -> None:
    """Add causal model arguments."""
    parser.add_argument(
        "--causal", action="store_true", help="Use GHCF-CE (causal head predicts non-BFI)"
    )
    parser.add_argument(
        "--ace",
        action="store_true",
        help="Use GHCF-ACE: GNN predicts all interactions and is ensembled with the "
        "causal head for non-BFI predictions",
    )
    parser.add_argument(
        "--ace_weight_mode",
        type=str,
        default=ACE_WEIGHT_MODE,
        choices=["learnable", "fixed"],
        help="GHCF-ACE GNN/causal mixing: 'learnable' per-type gate or 'fixed' weights",
    )
    parser.add_argument(
        "--ace_w_mlp",
        type=float,
        default=None,
        help="GHCF-ACE fixed-mode GNN weight applied to all non-BFI types (overrides config)",
    )
    parser.add_argument(
        "--ace_w_causal",
        type=float,
        default=None,
        help="GHCF-ACE fixed-mode causal weight applied to all non-BFI types (overrides config)",
    )
    parser.add_argument(
        "--freeze_sem",
        action=argparse.BooleanOptionalAction,
        default=FREEZE_SEM,
        help=(
            "Freeze the causal SEM head coefficients at their discovered LiNGAM "
            "values (the DAG mask is fixed either way). Default fine-tunes them; "
            "use --freeze_sem to keep the causal head as a pure structural prior."
        ),
    )
    parser.add_argument(
        "--causal_path",
        type=str,
        default=CAUSAL_MODEL_PATH,
        help="Path to causal model",
    )
    parser.add_argument(
        "--causal_config",
        type=str,
        default=CAUSAL_CONFIG_PATH,
        help="Path to causal config",
    )


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-related arguments."""
    parser.add_argument(
        "--k_values", type=str, default="[3,5,10]", help="Top-K values for evaluation"
    )


def add_cv_arguments(parser: argparse.ArgumentParser) -> None:
    """Add cross-validation arguments (defaults from configs/config.py)."""
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run k-fold cross-validation instead of a single train/test split",
    )
    parser.add_argument(
        "--n_folds", type=int, default=CV_CONFIG["n_folds"], help="Number of CV folds"
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=CV_CONFIG["stratified"],
        help="Use stratified CV (groups users by interaction count)",
    )
    parser.add_argument(
        "--cv_epochs", type=int, default=CV_CONFIG["cv_epochs"], help="Number of epochs per fold"
    )
    parser.add_argument("--adj_type", type=str, default=ADJ_TYPE, help="Adjacency matrix type")
    parser.add_argument(
        "--normalize_interactions",
        action=argparse.BooleanOptionalAction,
        default=NORMALIZE_INTERACTIONS,
        help=(
            "Normalize interaction targets (BFI MinMax; RATING/VQ/SF z-scored "
            "via the shared causal stats). Use --no-normalize_interactions to disable. "
            "Note: the causal models require this enabled for consistent losses."
        ),
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Optional tag prepended to the CV output directory name "
            "(checkpoints/cv_{run_tag}_{timestamp}/) so parallel sweep runs "
            "don't collide on second-resolution timestamps."
        ),
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with subcommands for train and evaluate.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = create_base_parser()
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train GHCF model")
    add_common_arguments(train_parser)
    add_training_arguments(train_parser)
    add_gradnorm_arguments(train_parser)
    add_causal_arguments(train_parser)
    add_evaluation_arguments(train_parser)
    add_cv_arguments(train_parser)

    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    add_common_arguments(eval_parser)
    eval_parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory containing trained model"
    )
    add_gradnorm_arguments(eval_parser)
    add_causal_arguments(eval_parser)
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_state.pth",
        choices=["best_state.pth", "last_state.pth"],
        help="Checkpoint to load",
    )
    add_evaluation_arguments(eval_parser)

    return parser.parse_args()
