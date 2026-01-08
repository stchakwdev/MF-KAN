"""MFKAN loss functions.

This module implements the loss functions for MFKAN training.

From the paper (Eq. 6):
    L_H = (1/N_HF) * Σ [K_H(x_j) - f_H(x_j)]^2 + λ_α * α^n + w * Σ ||Φ_nl||

where:
- First term: MSE between prediction and high-fidelity target
- Second term: Alpha penalty to encourage linear correlation
- Third term: Regularization on nonlinear network weights
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MFKANLoss(nn.Module):
    """Loss function for MFKAN high-fidelity training.

    The loss consists of three components:
    1. MSE loss between prediction and target
    2. Alpha penalty to encourage using linear correlation
    3. Regularization on nonlinear network weights

    Args:
        lambda_alpha: Penalty coefficient for alpha (default: 0.01).
        alpha_exponent: Exponent n for alpha penalty (default: 4).
        regularization_weight: Weight for spline regularization (default: 0.0).
        reduction: Reduction mode for MSE ('mean' or 'sum').
    """

    def __init__(
        self,
        lambda_alpha: float = 0.01,
        alpha_exponent: int = 4,
        regularization_weight: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.alpha_exponent = alpha_exponent
        self.regularization_weight = regularization_weight
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        alpha: torch.Tensor,
        regularization_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MFKAN loss.

        Args:
            y_pred: Predicted values from MFKAN.
            y_true: True high-fidelity values.
            alpha: Current alpha value (clamped to [0, 1]).
            regularization_loss: Regularization loss from nonlinear KAN.

        Returns:
            total_loss: Combined loss value.
            components: Dictionary of loss components for logging:
                - 'mse': MSE loss value
                - 'alpha_penalty': Alpha penalty value
                - 'regularization': Regularization loss value
                - 'alpha': Current alpha value
                - 'total': Total loss value
        """
        # MSE loss
        mse_loss = self.mse(y_pred, y_true)

        # Alpha penalty: λ_α * α^n
        # This encourages the model to use linear correlation when possible
        alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
        alpha_penalty = self.lambda_alpha * (alpha_clamped ** self.alpha_exponent)

        # Regularization on nonlinear network
        if regularization_loss is not None and self.regularization_weight > 0:
            reg_loss = self.regularization_weight * regularization_loss
        else:
            reg_loss = torch.tensor(0.0, device=y_pred.device)

        # Total loss
        total_loss = mse_loss + alpha_penalty + reg_loss

        # Component dictionary for logging
        components = {
            "mse": mse_loss.item(),
            "alpha_penalty": alpha_penalty.item(),
            "regularization": reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            "alpha": alpha_clamped.item(),
            "total": total_loss.item(),
        }

        return total_loss, components


class LFKANLoss(nn.Module):
    """Loss function for low-fidelity KAN training.

    Simple MSE loss for training the low-fidelity model.

    Args:
        reduction: Reduction mode for MSE ('mean' or 'sum').
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute low-fidelity loss.

        Args:
            y_pred: Predicted values from K_L.
            y_true: True low-fidelity values.

        Returns:
            loss: MSE loss value.
            components: Dictionary with loss components.
        """
        loss = self.mse(y_pred, y_true)
        components = {"mse": loss.item(), "total": loss.item()}
        return loss, components


class CombinedMFKANLoss(nn.Module):
    """Combined loss for joint low and high-fidelity training (optional).

    This can be used if you want to train both fidelity levels simultaneously,
    though the paper recommends sequential training.

    Args:
        lf_weight: Weight for low-fidelity loss.
        hf_weight: Weight for high-fidelity loss.
        lambda_alpha: Penalty coefficient for alpha.
        alpha_exponent: Exponent for alpha penalty.
        regularization_weight: Weight for regularization.
    """

    def __init__(
        self,
        lf_weight: float = 1.0,
        hf_weight: float = 1.0,
        lambda_alpha: float = 0.01,
        alpha_exponent: int = 4,
        regularization_weight: float = 0.0,
    ):
        super().__init__()
        self.lf_weight = lf_weight
        self.hf_weight = hf_weight
        self.lf_loss = LFKANLoss()
        self.hf_loss = MFKANLoss(
            lambda_alpha=lambda_alpha,
            alpha_exponent=alpha_exponent,
            regularization_weight=regularization_weight,
        )

    def forward(
        self,
        y_pred_lf: torch.Tensor,
        y_true_lf: torch.Tensor,
        y_pred_hf: torch.Tensor,
        y_true_hf: torch.Tensor,
        alpha: torch.Tensor,
        regularization_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            y_pred_lf: Low-fidelity predictions.
            y_true_lf: Low-fidelity targets.
            y_pred_hf: High-fidelity predictions.
            y_true_hf: High-fidelity targets.
            alpha: Current alpha value.
            regularization_loss: Regularization from nonlinear KAN.

        Returns:
            total_loss: Combined weighted loss.
            components: Dictionary with all loss components.
        """
        lf_loss, lf_components = self.lf_loss(y_pred_lf, y_true_lf)
        hf_loss, hf_components = self.hf_loss(
            y_pred_hf, y_true_hf, alpha, regularization_loss
        )

        total_loss = self.lf_weight * lf_loss + self.hf_weight * hf_loss

        components = {
            "lf_mse": lf_components["mse"],
            "hf_mse": hf_components["mse"],
            "alpha_penalty": hf_components["alpha_penalty"],
            "regularization": hf_components["regularization"],
            "alpha": hf_components["alpha"],
            "total": total_loss.item(),
        }

        return total_loss, components
