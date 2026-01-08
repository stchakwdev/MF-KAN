"""Multifidelity Kolmogorov-Arnold Network (MFKAN).

The MFKAN architecture consists of three blocks:
1. K_L (Low-Fidelity KAN): Learns a surrogate for low-fidelity data
2. K_l (Linear KAN): Learns linear correlation between LF output and HF data
3. K_nl (Nonlinear KAN): Learns nonlinear correction to the linear correlation

The high-fidelity prediction is:
    K_H(x) = α * K_nl(x, f_L(x)) + (1 - α) * K_l(x, f_L(x))

where α is a trainable blending parameter.

Reference:
    Howard et al. (2024) "Multifidelity Kolmogorov-Arnold Networks"
    arXiv:2410.14764
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mfkan.configs.config import MFKANConfig, KANConfig
from mfkan.models.kan_backend import KAN, KANLayer
from mfkan.models.linear_kan import LinearKAN


class MFKAN(nn.Module):
    """Multifidelity Kolmogorov-Arnold Network.

    This model learns to predict high-fidelity outputs using a combination of
    low-fidelity data (abundant but less accurate) and high-fidelity data
    (sparse but accurate).

    Architecture:
        Input x → K_L(x) → f_L
        (x, f_L) → K_l → linear correction
        (x, f_L) → K_nl → nonlinear correction
        Output: α * nonlinear + (1-α) * linear

    Args:
        config: MFKAN configuration.

    Attributes:
        k_lf: Low-fidelity KAN network.
        k_linear: Linear KAN for linear correlation.
        k_nonlinear: Nonlinear KAN for residual learning.
        alpha: Trainable blending parameter in [0, 1].
    """

    def __init__(self, config: MFKANConfig):
        super().__init__()
        self.config = config

        # K_L: Low-fidelity KAN
        lf_config = config.get_lf_config()
        self.k_lf = KAN(lf_config, backend=config.backend)

        # K_l: Linear KAN
        # Input: concatenation of x and f_L(x)
        linear_input_dim = config.input_dim + config.output_dim
        self.k_linear = LinearKAN(
            input_dim=linear_input_dim,
            output_dim=config.output_dim,
            grid_range=config.grid_range,
        )

        # K_nl: Nonlinear KAN
        nl_config = config.get_nl_config()
        self.k_nonlinear = KAN(nl_config, backend=config.backend)

        # Alpha blending parameter
        # Initialize with alpha_init, constrained to [0, 1] during forward
        self._alpha = nn.Parameter(
            torch.tensor(config.alpha_init),
            requires_grad=config.alpha_trainable,
        )

        # Track training state
        self._lf_trained = False
        self._lf_frozen = False

    @property
    def alpha(self) -> torch.Tensor:
        """Get clamped alpha value in [0, 1]."""
        return torch.clamp(self._alpha, 0.0, 1.0)

    @property
    def alpha_raw(self) -> torch.Tensor:
        """Get raw (unclamped) alpha parameter."""
        return self._alpha

    def freeze_low_fidelity(self) -> None:
        """Freeze K_L parameters after low-fidelity training.

        This should be called after training the low-fidelity model
        and before training the high-fidelity components.
        """
        for param in self.k_lf.parameters():
            param.requires_grad = False
        self._lf_frozen = True

    def unfreeze_low_fidelity(self) -> None:
        """Unfreeze K_L parameters for fine-tuning (optional)."""
        for param in self.k_lf.parameters():
            param.requires_grad = True
        self._lf_frozen = False

    def mark_lf_trained(self) -> None:
        """Mark the low-fidelity model as trained."""
        self._lf_trained = True

    @property
    def is_lf_trained(self) -> bool:
        """Check if low-fidelity model has been trained."""
        return self._lf_trained

    @property
    def is_lf_frozen(self) -> bool:
        """Check if low-fidelity model is frozen."""
        return self._lf_frozen

    def forward_lf(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through low-fidelity model only.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Low-fidelity prediction of shape (..., output_dim)
        """
        return self.k_lf(x)

    def forward_hf(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for high-fidelity prediction.

        Args:
            x: Input tensor of shape (..., input_dim)
            return_components: If True, also return intermediate values.

        Returns:
            High-fidelity prediction of shape (..., output_dim)
            If return_components=True, also returns dict with:
                - 'y_lf': Low-fidelity prediction
                - 'y_linear': Linear correction
                - 'y_nonlinear': Nonlinear correction
                - 'alpha': Current alpha value
        """
        # Get low-fidelity prediction
        if self._lf_frozen:
            with torch.no_grad():
                y_lf = self.k_lf(x)
        else:
            y_lf = self.k_lf(x)

        # Concatenate input with LF prediction for correction networks
        # This follows the paper's approach where K_l and K_nl take
        # both x and f_L(x) as input
        x_concat = torch.cat([x, y_lf], dim=-1)

        # Linear and nonlinear corrections
        y_linear = self.k_linear(x_concat)
        y_nonlinear = self.k_nonlinear(x_concat)

        # Blend corrections using alpha
        alpha = self.alpha
        y_correction = alpha * y_nonlinear + (1 - alpha) * y_linear

        # High-fidelity output is LF prediction plus correction
        y_hf = y_lf + y_correction

        if return_components:
            components = {
                "y_lf": y_lf,
                "y_linear": y_linear,
                "y_nonlinear": y_nonlinear,
                "alpha": alpha,
            }
            return y_hf, components

        return y_hf

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "auto",
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim)
            mode: "auto" (use HF if LF trained), "lf", or "hf"
            return_components: If True, return intermediate values (HF mode only)

        Returns:
            Prediction tensor of shape (..., output_dim)
        """
        if mode == "lf":
            return self.forward_lf(x)
        elif mode == "hf":
            return self.forward_hf(x, return_components=return_components)
        else:  # auto
            if self._lf_trained:
                return self.forward_hf(x, return_components=return_components)
            return self.forward_lf(x)

    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss from nonlinear KAN.

        From the paper (Eq. 7):
            ||Φ_nl|| = (1 / n_in * n_out) * Σ |φ^nl_{i,j}|^2

        Returns:
            Regularization loss value.
        """
        return self.k_nonlinear.regularization_loss()

    def get_trainable_hf_params(self) -> list:
        """Get list of parameters to train for high-fidelity phase.

        Returns:
            List of parameters (K_l, K_nl, and alpha).
        """
        params = []
        params.extend(self.k_linear.parameters())
        params.extend(self.k_nonlinear.parameters())
        if self.config.alpha_trainable:
            params.append(self._alpha)
        return params

    def get_trainable_lf_params(self) -> list:
        """Get list of parameters to train for low-fidelity phase.

        Returns:
            List of K_L parameters.
        """
        return list(self.k_lf.parameters())

    def state_dict_info(self) -> Dict[str, any]:
        """Get information about model state.

        Returns:
            Dictionary with model state information.
        """
        return {
            "lf_trained": self._lf_trained,
            "lf_frozen": self._lf_frozen,
            "alpha": self.alpha.item(),
            "config": {
                "input_dim": self.config.input_dim,
                "output_dim": self.config.output_dim,
                "backend": self.config.backend,
            },
        }

    def __repr__(self) -> str:
        return (
            f"MFKAN(\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  output_dim={self.config.output_dim},\n"
            f"  backend='{self.config.backend}',\n"
            f"  alpha={self.alpha.item():.4f},\n"
            f"  lf_trained={self._lf_trained},\n"
            f"  lf_frozen={self._lf_frozen}\n"
            f")"
        )
