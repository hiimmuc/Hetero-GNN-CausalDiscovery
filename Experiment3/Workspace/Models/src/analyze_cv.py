"""
Analyze cross-validation results.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_cv_results(cv_dir: str):
    """Load cross-validation results."""
    cv_path = Path(cv_dir)

    with open(cv_path / "cv_results.json", "r") as f:
        results = json.load(f)

    return results


def plot_cv_metrics(results, save_dir: Path):
    """Plot cross-validation metrics."""
    try:
        import matplotlib.pyplot as plt

        fold_results = results["fold_results"]
        n_folds = len(fold_results)

        # Get all metrics
        metrics = list(fold_results[0]["best_metrics"].keys())

        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for i, metric in enumerate(metrics):
            values = [fold["best_metrics"][metric] for fold in fold_results]

            ax = axes[i]
            ax.bar(range(1, n_folds + 1), values, alpha=0.7)
            ax.axhline(
                y=np.mean(values), color="r", linestyle="--", label=f"Mean: {np.mean(values):.4f}"
            )
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


def create_cv_report(results, save_dir: Path):
    """Create comprehensive CV report."""
    report = []

    report.append("# Cross-Validation Report\n\n")
    report.append(f"**Number of folds:** {results['n_folds']}\n")
    report.append(f"**Total CV time:** {results['cv_time']:.2f}s\n")
    report.append(f"**Best fold:** {results['best_fold'] + 1}\n")
    report.append(f"**Worst fold:** {results['worst_fold'] + 1}\n\n")

    report.append("## Mean ± Std Results\n\n")
    report.append("| Metric | Mean | Std |\n")
    report.append("|--------|------|-----|\n")

    for metric, mean_val in results["mean_metrics"].items():
        std_val = results["std_metrics"][metric]
        report.append(f"| {metric} | {mean_val:.4f} | {std_val:.4f} |\n")

    report.append("\n## Per-Fold Results\n\n")
    report.append(
        "| Fold | " + " | ".join(results["mean_metrics"].keys()) + " | Training Time |\n"
    )
    report.append("|------|" + "|".join(["-" * 10] * (len(results["mean_metrics"]) + 1)) + "|\n")

    for i, fold_result in enumerate(results["fold_results"]):
        row = f"| {i+1} |"
        for metric in results["mean_metrics"].keys():
            row += f" {fold_result['best_metrics'][metric]:.4f} |"
        row += f" {fold_result['training_time']:.2f}s |\n"
        report.append(row)

    with open(save_dir / "cv_report.md", "w") as f:
        f.writelines(report)

    print("CV report saved.")


def print_cv_summary(results):
    """Print summary of CV results."""
    print("\nCV Analysis Summary:")
    print("-" * 40)
    for metric, mean_val in results["mean_metrics"].items():
        std_val = results["std_metrics"][metric]
        print(f"{metric:15s}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"\nBest fold: {results['best_fold'] + 1}")
    print(f"Worst fold: {results['worst_fold'] + 1}")
    print(f"Total CV time: {results['cv_time']:.2f}s")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze CV results")
    parser.add_argument(
        "--cv_dir", type=str, required=True, help="Directory containing CV results"
    )

    args = parser.parse_args()

    # Load results
    results = load_cv_results(args.cv_dir)
    cv_dir = Path(args.cv_dir)

    # Create plots
    print("Creating plots...")
    plot_cv_metrics(results, cv_dir)

    # Create report
    print("Creating report...")
    create_cv_report(results, cv_dir)

    # Print summary
    print_cv_summary(results)

    print(f"\nAnalysis saved to: {cv_dir}")


if __name__ == "__main__":
    main()
