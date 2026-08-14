"""
Entry point to run the LiNGAM causal discovery pipeline.

Run from the `Models/` project root:

    python src/run_causal_discovery.py
    python src/run_causal_discovery.py --config src/configs/causal_config.yaml
"""

import argparse
import logging

from models.CausalModel import CausalModel

logger = logging.getLogger(__name__)


def main() -> None:
    """Load data, fit the LiNGAM model, run bootstrap analysis, and save outputs."""
    parser = argparse.ArgumentParser(description="Run LiNGAM causal discovery")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/causal_config.yaml",
        help="Path to the causal model YAML config",
    )
    args = parser.parse_args()

    causal_model = CausalModel(args.config)
    causal_model.run_complete_analysis()

    # Print comprehensive model information including system of equations
    causal_model.print_model_info()

    # print("\n" + "=" * 60)
    # print("SEPARATE SYSTEM OF EQUATIONS CALLS:")
    # causal_model.print_system_of_equations(use_robust=False)
    # if hasattr(causal_model, "robust_adjacency_matrix"):
    #     causal_model.print_system_of_equations(use_robust=True)

    if causal_model.config["inference"]["enable_inference"]:
        example_input = {
            "Difference BFI-TIPI O": 0.5,
            "Difference BFI-TIPI C": -0.2,
            "Difference BFI-TIPI E": 0.1,
            "Difference BFI-TIPI A": 0.3,
            "Difference BFI-TIPI N": -0.1,
        }
        predictions = causal_model.predict(example_input)
        logger.info(f"Example prediction: {predictions}")


if __name__ == "__main__":
    main()
