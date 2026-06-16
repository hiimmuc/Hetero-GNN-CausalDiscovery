"""
Main entry point for GHCF eHMI recommendation system.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utility.parser import parse_args


def main() -> None:
    """Main entry point with command selection."""
    args = parse_args()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.command == "train":
        if getattr(args, "cv", False):
            print("Starting GHCF cross-validation training...")
            from cross_validation import CrossValidationTrainer

            cv_trainer = CrossValidationTrainer(
                args, n_folds=args.n_folds, stratified=args.stratified
            )
            cv_trainer.run()
        else:
            print("Starting GHCF model training...")
            from train import Trainer

            trainer = Trainer(args)
            trainer.run()

    elif args.command == "evaluate":
        print("Starting GHCF model evaluation...")
        from evaluate import Evaluator

        evaluator = Evaluator(args)
        evaluator.run()

    else:
        print("Error: Please specify a command (train or evaluate)")
        print("Use --help for more information")
        sys.exit(1)


if __name__ == "__main__":
    main()
