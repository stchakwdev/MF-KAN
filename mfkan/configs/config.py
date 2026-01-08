"""Configuration dataclasses for MFKAN.

This module provides configuration classes for KAN networks and MFKAN training.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class KANConfig:
    """Configuration for individual KAN networks.

    Attributes:
        input_dim: Number of input features.
        output_dim: Number of output features.
        hidden_dims: List of hidden layer dimensions.
        grid_size: Number of grid intervals for B-splines (default: 5).
        spline_order: Polynomial degree of B-splines (default: 3).
        scale_noise: Noise scale for weight initialization (default: 0.1).
        scale_base: Scale for base weight (default: 1.0).
        scale_spline: Scale for spline weight (default: 1.0).
        grid_range: Range for grid points (default: (-1.0, 1.0)).
        base_activation: Base activation function name (default: "silu").
    """

    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=list)
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    grid_range: Tuple[float, float] = (-1.0, 1.0)
    base_activation: str = "silu"

    def get_layer_dims(self) -> List[int]:
        """Get full layer dimensions including input and output."""
        return [self.input_dim] + list(self.hidden_dims) + [self.output_dim]


@dataclass
class MFKANConfig:
    """Configuration for Multifidelity KAN model.

    The MFKAN consists of three blocks:
    - K_L: Low-fidelity KAN (pretrained, then frozen)
    - K_l: Linear KAN for linear correlation
    - K_nl: Nonlinear KAN for nonlinear residual

    Attributes:
        input_dim: Number of input features.
        output_dim: Number of output features.
        backend: KAN backend to use ("faster" or "efficient").

        # Low-fidelity KAN (K_L)
        lf_hidden_dims: Hidden layer dimensions for low-fidelity KAN.
        lf_grid_size: Grid size for low-fidelity KAN.
        lf_spline_order: Spline order for low-fidelity KAN.

        # Nonlinear KAN (K_nl)
        nl_hidden_dims: Hidden layer dimensions for nonlinear KAN.
        nl_grid_size: Grid size for nonlinear KAN.
        nl_spline_order: Spline order for nonlinear KAN.

        # Alpha blending
        alpha_init: Initial value for alpha (default: 0.5).
        alpha_trainable: Whether alpha is trainable (default: True).

        # Loss parameters
        lambda_alpha: Penalty coefficient for alpha (default: 0.01).
        alpha_exponent: Exponent n for alpha penalty (default: 4).
        regularization_weight: Weight for spline regularization (default: 0.0).
    """

    input_dim: int
    output_dim: int
    backend: Literal["faster", "efficient"] = "faster"

    # Low-fidelity KAN (K_L)
    lf_hidden_dims: List[int] = field(default_factory=lambda: [32, 32])
    lf_grid_size: int = 5
    lf_spline_order: int = 3

    # Nonlinear KAN (K_nl)
    nl_hidden_dims: List[int] = field(default_factory=lambda: [16, 16])
    nl_grid_size: int = 5
    nl_spline_order: int = 3

    # Alpha blending parameter
    alpha_init: float = 0.5
    alpha_trainable: bool = True

    # Loss parameters
    lambda_alpha: float = 0.01
    alpha_exponent: int = 4
    regularization_weight: float = 0.0

    # Grid range
    grid_range: Tuple[float, float] = (-1.0, 1.0)

    def get_lf_config(self) -> KANConfig:
        """Get configuration for low-fidelity KAN."""
        return KANConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.lf_hidden_dims,
            grid_size=self.lf_grid_size,
            spline_order=self.lf_spline_order,
            grid_range=self.grid_range,
        )

    def get_nl_config(self) -> KANConfig:
        """Get configuration for nonlinear KAN."""
        return KANConfig(
            input_dim=self.input_dim + self.output_dim,  # x + f_L(x)
            output_dim=self.output_dim,
            hidden_dims=self.nl_hidden_dims,
            grid_size=self.nl_grid_size,
            spline_order=self.nl_spline_order,
            grid_range=self.grid_range,
        )


@dataclass
class TrainingConfig:
    """Configuration for MFKAN training.

    Attributes:
        # Low-fidelity training
        lf_epochs: Number of epochs for low-fidelity training.
        lf_lr: Learning rate for low-fidelity training.
        lf_batch_size: Batch size for low-fidelity training.
        lf_grid_update_freq: Frequency of grid updates (0 to disable).

        # High-fidelity training
        hf_epochs: Number of epochs for high-fidelity training.
        hf_lr: Learning rate for high-fidelity training.
        hf_batch_size: Batch size for high-fidelity training.

        # Common settings
        weight_decay: Weight decay for optimizer.
        early_stopping_patience: Patience for early stopping (0 to disable).
        checkpoint_dir: Directory for saving checkpoints.
        device: Device to use for training ("cuda", "cpu", "auto").
        seed: Random seed for reproducibility.
    """

    # Low-fidelity training
    lf_epochs: int = 1000
    lf_lr: float = 1e-3
    lf_batch_size: int = 64
    lf_grid_update_freq: int = 0  # Disabled by default

    # High-fidelity training
    hf_epochs: int = 500
    hf_lr: float = 1e-3
    hf_batch_size: int = 32
    hf_grid_update_freq: int = 0  # Disabled by default

    # Common settings
    weight_decay: float = 0.0
    early_stopping_patience: int = 0  # Disabled by default
    checkpoint_dir: Optional[str] = None
    device: str = "auto"
    seed: int = 42
    verbose: bool = True
    log_interval: int = 100

    def get_device(self) -> str:
        """Get the actual device to use."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
