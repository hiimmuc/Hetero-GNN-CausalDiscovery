"""
GradNorm implementation for dynamic multi-task loss balancing.

Based on "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
by Chen et al. (2018).
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class GradNorm:
    """
    GradNorm implementation for automatic loss balancing in multi-task learning.

    This class dynamically adjusts task weights based on the relative training rates
    of different tasks to ensure balanced learning across all tasks.
    """

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
        initial_task_weights: Dict[str, float] = None,
        device: str = "cpu",
    ):
        """
        Initialize GradNorm.

        Args:
            task_names: List of task names (interaction types)
            alpha: Restoring force hyperparameter (typically 0.12-3.0)
                  Higher values = stronger restoring force towards equal learning rates
            initial_task_weights: Initial weights for each task
            device: Device to run computations on
        """
        self.task_names = task_names
        self.alpha = alpha
        self.device = device
        self.num_tasks = len(task_names)

        # Initialize task weights
        if initial_task_weights is None:
            self.task_weights = {task: 1.0 for task in task_names}
        else:
            self.task_weights = initial_task_weights.copy()

        # Convert to tensor for gradient computation
        self.task_weight_tensor = nn.Parameter(
            torch.tensor(
                [self.task_weights[task] for task in task_names],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
        )

        # Track initial losses for relative loss computation
        self.initial_losses = None
        self.step_count = 0

    def compute_gradnorm_weights(
        self,
        shared_parameters: List[nn.Parameter],
        task_losses: Dict[str, torch.Tensor],
        task_gradients: Dict[str, List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute updated task weights using GradNorm algorithm.

        Args:
            shared_parameters: List of shared model parameters
            task_losses: Dictionary of task losses {task_name: loss_tensor}
            task_gradients: Pre-computed gradients (optional, will compute if None)

        Returns:
            Updated task weights dictionary
        """
        self.step_count += 1

        # Store initial losses for relative loss computation
        if self.initial_losses is None:
            self.initial_losses = {task: loss.item() for task, loss in task_losses.items()}
            return self.task_weights

        # Compute task gradients if not provided
        if task_gradients is None:
            task_gradients = self._compute_task_gradients(shared_parameters, task_losses)

        # Compute gradient norms for each task
        grad_norms = {}
        for task in self.task_names:
            if task in task_gradients and len(task_gradients[task]) > 0:
                grad_norm_squared = 0.0
                for grad in task_gradients[task]:
                    if grad is not None:
                        grad_norm_squared += torch.sum(grad**2).item()
                grad_norms[task] = np.sqrt(max(grad_norm_squared, 1e-8))
            else:
                grad_norms[task] = 1e-8

        # Compute relative loss rates
        relative_losses = {}
        for task in self.task_names:
            current_loss = task_losses[task].item()
            initial_loss = self.initial_losses[task]
            relative_losses[task] = current_loss / max(initial_loss, 1e-8)

        # Compute average relative loss rate and gradient norm
        avg_relative_loss = np.mean(list(relative_losses.values()))
        avg_grad_norm = np.mean(list(grad_norms.values()))

        # Compute target gradient norms
        target_grad_norms = {}
        for task in self.task_names:
            r_i = relative_losses[task]
            # Target gradient norm: G_avg * (r_i / r_avg)^alpha
            ratio = abs(r_i / max(abs(avg_relative_loss), 1e-8)) ** self.alpha
            target_grad_norms[task] = avg_grad_norm * ratio

        # Compute weight updates using GradNorm principle
        # L_gradnorm = sum_i |G_i(t) / avg(G(t)) * (r_i / avg(r))^alpha - 1|
        weight_updates = {}
        for i, task in enumerate(self.task_names):
            target_norm = max(target_grad_norms[task], 1e-8)
            current_norm = max(grad_norms[task], 1e-8)

            # Compute the gradient norm ratio error
            ratio_error = abs(current_norm / target_norm - 1.0)

            # Update weights to reduce this error
            # If current norm is too high compared to target, reduce weight
            # If current norm is too low compared to target, increase weight
            current_weight = self.task_weights[task]
            if current_norm > target_norm:
                # Gradient norm too high -> reduce weight
                weight_updates[task] = current_weight * (1.0 - 0.025 * ratio_error)
            else:
                # Gradient norm too low -> increase weight
                weight_updates[task] = current_weight * (1.0 + 0.025 * ratio_error)

            # Ensure positive weights
            weight_updates[task] = max(weight_updates[task], 0.1)

        # Renormalize weights to sum to number of tasks
        total_weight = sum(weight_updates.values())
        for task in self.task_names:
            weight_updates[task] = weight_updates[task] * self.num_tasks / total_weight

        # Update task weights
        self.task_weights.update(weight_updates)

        # Update tensor weights for consistency
        with torch.no_grad():
            for i, task in enumerate(self.task_names):
                self.task_weight_tensor[i] = self.task_weights[task]

        return self.task_weights.copy()

    def _compute_task_gradients(
        self, shared_parameters: List[nn.Parameter], task_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Compute gradients for each task loss with respect to shared parameters.

        Args:
            shared_parameters: List of shared model parameters
            task_losses: Dictionary of task losses

        Returns:
            Dictionary mapping task names to lists of gradients
        """
        task_gradients = {}

        for task, loss in task_losses.items():
            # Compute gradients for this task
            grads = torch.autograd.grad(
                loss, shared_parameters, retain_graph=True, allow_unused=True
            )
            task_gradients[task] = [g for g in grads if g is not None]

        return task_gradients

    def get_current_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        return self.task_weights.copy()

    def get_weights_tensor(self) -> torch.Tensor:
        """Get current weights as tensor."""
        return self.task_weight_tensor.detach().clone()

    def reset(self):
        """Reset GradNorm state."""
        self.initial_losses = None
        self.step_count = 0
        with torch.no_grad():
            self.task_weight_tensor.fill_(1.0)
        self.task_weights = {task: 1.0 for task in self.task_names}

    def compute_weighted_loss(
        self, task_losses: Dict[str, torch.Tensor], shared_parameters: List[nn.Parameter] = None
    ) -> torch.Tensor:
        """
        Compute weighted loss using current task weights and optionally update weights.

        Args:
            task_losses: Dictionary of task losses {task_name: loss_tensor}
            shared_parameters: Optional list of shared model parameters for GradNorm updates

        Returns:
            Total weighted loss
        """
        # Update weights using GradNorm if we have enough history and shared parameters
        if (
            self.initial_losses is not None
            and self.step_count > 0
            and shared_parameters is not None
            and len(shared_parameters) > 0
        ):
            try:
                updated_weights = self.compute_gradnorm_weights(shared_parameters, task_losses)
                self.task_weights.update(updated_weights)
            except (RuntimeError, AttributeError, TypeError) as e:
                # Fall back to current weights if gradient computation fails
                print(f"GradNorm weight update failed: {e}. Using current weights.")
                pass

        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for task, loss in task_losses.items():
            if task in self.task_weights:
                total_loss = total_loss + self.task_weights[task] * loss
            else:
                total_loss = total_loss + loss  # Default weight of 1.0

        return total_loss


def compute_weighted_loss(
    individual_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    use_gradnorm: bool = False,
    gradnorm_handler: GradNorm = None,
    shared_parameters: List[nn.Parameter] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute weighted multi-task loss with optional GradNorm balancing.

    Args:
        individual_losses: Dictionary of individual task losses
        weights: Current task weights
        use_gradnorm: Whether to use GradNorm for dynamic balancing
        gradnorm_handler: GradNorm instance (required if use_gradnorm=True)
        shared_parameters: Shared model parameters (required if use_gradnorm=True)

    Returns:
        Tuple of (total_weighted_loss, updated_weights)
    """
    if use_gradnorm and gradnorm_handler is not None and shared_parameters is not None:
        # Update weights using GradNorm
        updated_weights = gradnorm_handler.compute_gradnorm_weights(
            shared_parameters, individual_losses
        )
    else:
        # Use static weights
        updated_weights = weights.copy()

    # Compute weighted total loss
    total_loss = 0.0
    for task, loss in individual_losses.items():
        if task in updated_weights:
            total_loss += updated_weights[task] * loss
        else:
            total_loss += loss  # Default weight of 1.0

    return total_loss, updated_weights


if __name__ == "__main__":
    # Test GradNorm implementation
    print("Testing GradNorm implementation...")

    # Test parameters
    task_names = ["RATING", "BFI_PERCEPTION", "VOICE_QUALITY"]
    device = "cpu"

    # Create GradNorm instance
    gradnorm = GradNorm(task_names, alpha=1.5, device=device)

    # Create dummy shared parameters
    shared_params = [nn.Parameter(torch.randn(10, 5, requires_grad=True))]

    # Create dummy task losses that depend on shared parameters
    # This ensures they have grad_fn and are part of the computation graph
    dummy_output = torch.sum(shared_params[0])
    task_losses = {
        "RATING": dummy_output * 2.5 + torch.tensor(0.1, requires_grad=True),
        "BFI_PERCEPTION": dummy_output * 1.8 + torch.tensor(0.1, requires_grad=True),
        "VOICE_QUALITY": dummy_output * 3.2 + torch.tensor(0.1, requires_grad=True),
    }

    print("Initial weights:", gradnorm.get_current_weights())

    # First call - should just store initial losses
    updated_weights = gradnorm.compute_gradnorm_weights(shared_params, task_losses)
    print("Updated weights (first call):", updated_weights)

    # Second call - should actually update weights
    updated_weights = gradnorm.compute_gradnorm_weights(shared_params, task_losses)
    print("Updated weights (second call):", updated_weights)

    # Test weighted loss computation
    total_loss, final_weights = compute_weighted_loss(
        task_losses,
        updated_weights,
        use_gradnorm=False,  # Test static weights first
        gradnorm_handler=gradnorm,
        shared_parameters=shared_params,
    )

    print(f"Total weighted loss (static): {total_loss.item():.4f}")
    print("Final weights (static):", final_weights)

    # Test with dynamic GradNorm
    total_loss_dynamic, final_weights_dynamic = compute_weighted_loss(
        task_losses,
        updated_weights,
        use_gradnorm=True,  # Test dynamic weights
        gradnorm_handler=gradnorm,
        shared_parameters=shared_params,
    )

    print(f"Total weighted loss (dynamic): {total_loss_dynamic.item():.4f}")
    print("Final weights (dynamic):", final_weights_dynamic)
    print("✓ GradNorm test completed")
