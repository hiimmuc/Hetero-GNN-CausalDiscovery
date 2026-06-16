"""
Plotting utilities for training visualization.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np


def save_loss_plots(loss_history: Dict[str, List[float]], output_dir: Path) -> None:
    """
    Save loss plots for training visualization.

    Args:
        loss_history: Dictionary containing loss history for each interaction type
        output_dir: Directory to save plots
    """
    if not loss_history:
        return

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot total loss
    if "total" in loss_history and loss_history["total"]:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history["total"], label="Total Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "total_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot individual interaction losses
    interaction_losses = {k: v for k, v in loss_history.items() if k != "total" and v}

    if interaction_losses:
        plt.figure(figsize=(12, 8))

        for interaction_type, losses in interaction_losses.items():
            if losses:  # Only plot if there are losses
                plt.plot(losses, label=f"{interaction_type} Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Individual Interaction Type Losses")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "individual_losses.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Create subplot for each interaction type
        n_interactions = len(interaction_losses)
        if n_interactions > 1:
            fig, axes = plt.subplots(
                nrows=(n_interactions + 1) // 2,
                ncols=2,
                figsize=(15, 5 * ((n_interactions + 1) // 2)),
            )

            if n_interactions == 1:
                axes = [axes]
            elif n_interactions <= 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for idx, (interaction_type, losses) in enumerate(interaction_losses.items()):
                if losses and idx < len(axes):
                    axes[idx].plot(losses, linewidth=2, color=f"C{idx}")
                    axes[idx].set_title(f"{interaction_type} Loss")
                    axes[idx].set_xlabel("Epoch")
                    axes[idx].set_ylabel("Loss")
                    axes[idx].grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(len(interaction_losses), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(plots_dir / "individual_losses_subplots.png", dpi=300, bbox_inches="tight")
            plt.close()


def save_weight_plots(weight_history: Dict[str, List[float]], output_dir: Path) -> None:
    """
    Save weight plots for GradNorm visualization.

    Args:
        weight_history: Dictionary containing weight history for each interaction type
        output_dir: Directory to save plots
    """
    if not weight_history:
        return

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Filter out empty weight histories
    filtered_weights = {k: v for k, v in weight_history.items() if v}

    if not filtered_weights:
        return

    # Plot all weights together
    plt.figure(figsize=(12, 8))

    for interaction_type, weights in filtered_weights.items():
        if weights:  # Only plot if there are weights
            plt.plot(
                weights, label=f"{interaction_type} Weight", linewidth=2, marker="o", markersize=3
            )

    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.title("GradNorm Weight Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "gradnorm_weights.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create subplot for each interaction type weight
    n_interactions = len(filtered_weights)
    if n_interactions > 1:
        fig, axes = plt.subplots(
            nrows=(n_interactions + 1) // 2, ncols=2, figsize=(15, 5 * ((n_interactions + 1) // 2))
        )

        if n_interactions == 1:
            axes = [axes]
        elif n_interactions <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, (interaction_type, weights) in enumerate(filtered_weights.items()):
            if weights and idx < len(axes):
                axes[idx].plot(weights, linewidth=2, marker="o", markersize=3, color=f"C{idx}")
                axes[idx].set_title(f"{interaction_type} Weight")
                axes[idx].set_xlabel("Epoch")
                axes[idx].set_ylabel("Weight")
                axes[idx].grid(True, alpha=0.3)

                # Add horizontal line at weight=1.0 for reference
                axes[idx].axhline(
                    y=1.0, color="red", linestyle="--", alpha=0.5, label="Initial Weight"
                )
                axes[idx].legend()

        # Hide unused subplots
        for idx in range(len(filtered_weights), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(plots_dir / "gradnorm_weights_subplots.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Save weight statistics
    weight_stats_file = plots_dir / "weight_statistics.txt"
    with open(weight_stats_file, "w") as f:
        f.write("GradNorm Weight Statistics\n")
        f.write("=" * 30 + "\n\n")

        for interaction_type, weights in filtered_weights.items():
            if weights:
                weights_array = np.array(weights)
                f.write(f"{interaction_type}:\n")
                f.write(f"  Initial weight: {weights[0]:.4f}\n")
                f.write(f"  Final weight: {weights[-1]:.4f}\n")
                f.write(f"  Mean weight: {weights_array.mean():.4f}\n")
                f.write(f"  Std weight: {weights_array.std():.4f}\n")
                f.write(f"  Min weight: {weights_array.min():.4f}\n")
                f.write(f"  Max weight: {weights_array.max():.4f}\n")
                f.write(f"  Weight change: {weights[-1] - weights[0]:.4f}\n")
                f.write("\n")


def create_training_summary_plot(
    loss_history: Dict[str, List[float]], weight_history: Dict[str, List[float]], output_dir: Path
) -> None:
    """
    Create a comprehensive training summary plot.

    Args:
        loss_history: Dictionary containing loss history
        weight_history: Dictionary containing weight history
        output_dir: Directory to save plots
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Create a comprehensive summary plot
    plt.figure(figsize=(20, 12))

    # Plot 1: Total loss
    if "total" in loss_history and loss_history["total"]:
        plt.subplot(2, 3, 1)
        plt.plot(loss_history["total"], linewidth=2, color="blue")
        plt.title("Total Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

    # Plot 2: Individual losses
    interaction_losses = {k: v for k, v in loss_history.items() if k != "total" and v}
    if interaction_losses:
        plt.subplot(2, 3, 2)
        for interaction_type, losses in interaction_losses.items():
            if losses:
                plt.plot(losses, label=interaction_type, linewidth=2)
        plt.title("Individual Interaction Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: Weight evolution (if available)
    filtered_weights = {k: v for k, v in weight_history.items() if v}
    if filtered_weights:
        plt.subplot(2, 3, 3)
        for interaction_type, weights in filtered_weights.items():
            if weights:
                plt.plot(weights, label=interaction_type, linewidth=2, marker="o", markersize=2)
        plt.title("GradNorm Weight Evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Weight")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)

    # Plot 4-6: Individual loss trends (top 3 interaction types)
    sorted_interactions = sorted(
        interaction_losses.items(), key=lambda x: len(x[1]), reverse=True
    )[:3]
    for idx, (interaction_type, losses) in enumerate(sorted_interactions):
        if losses:
            plt.subplot(2, 3, 4 + idx)
            plt.plot(losses, linewidth=2, color=f"C{idx}")
            plt.title(f"{interaction_type} Loss Trend")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "training_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
