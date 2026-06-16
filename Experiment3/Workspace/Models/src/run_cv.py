"""
Script to run cross-validation experiments for GHCF models.
"""

import argparse

from configs.config import CAUSAL_CONFIG_PATH, CAUSAL_MODEL_PATH
from cross_validation import CrossValidationTrainer


def main():
    """Main function to run cross-validation."""
    parser = argparse.ArgumentParser(description="Run GHCF Cross-Validation")

    # Model arguments
    parser.add_argument(
        "--interactions",
        type=str,
        nargs="+",
        default=["RATING", "BFI_PERCEPTION", "VOICE_QUALITY", "SUBJECTIVE_FEELING"],
        help="Interaction types to use",
    )
    parser.add_argument("--k_values", type=str, default="[3,5,10]", help="K values for evaluation")

    # Cross-validation arguments
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--stratified", action="store_true", default=True, help="Use stratified CV"
    )
    parser.add_argument("--cv_epochs", type=int, default=100, help="Number of epochs per fold")

    # Model type arguments
    parser.add_argument("--causal", action="store_true", help="Use GHCF_Causal model")
    parser.add_argument("--refine", action="store_true", help="Use GHCF_Refine model")
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

    # Training arguments (shared definitions with the main parser)
    from utility.parser import add_gradnorm_arguments, add_training_arguments

    add_training_arguments(parser)
    add_gradnorm_arguments(parser)
    parser.add_argument("--adj_type", type=str, default="plain", help="Adjacency matrix type")
    parser.add_argument(
        "--normalize_interactions", action="store_true", help="Normalize interaction values"
    )

    args = parser.parse_args()

    # Run cross-validation
    cv_trainer = CrossValidationTrainer(args, n_folds=args.n_folds, stratified=args.stratified)

    results = cv_trainer.run()

    print("\nCross-validation completed!")
    print(f"Results saved to: {cv_trainer.cv_dir}")

    return results


if __name__ == "__main__":
    main()
