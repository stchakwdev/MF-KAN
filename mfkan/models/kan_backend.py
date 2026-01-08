"""KAN backend abstraction layer.

This module provides a unified interface for different KAN implementations:
- faster-kan: 3.33x faster than efficient-kan, uses RSWAF basis functions
- efficient-kan: Well-tested B-spline implementation

If no external KAN library is available, falls back to a pure PyTorch implementation.
"""

from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mfkan.configs.config import KANConfig


# Try to import external KAN libraries
_FASTER_KAN_AVAILABLE = False
_EFFICIENT_KAN_AVAILABLE = False

try:
    from fasterkan import FasterKAN as _FasterKAN
    from fasterkan import FasterKANLayer as _FasterKANLayer

    _FASTER_KAN_AVAILABLE = True
except ImportError:
    pass

try:
    from efficient_kan import KAN as _EfficientKAN
    from efficient_kan import KANLinear as _EfficientKANLinear

    _EFFICIENT_KAN_AVAILABLE = True
except ImportError:
    pass


class RSWAFBasis(nn.Module):
    """Reflectional Switch Activation Function (RSWAF) basis.

    RSWAF approximates B-splines using:
        b_i(u) = 1 - tanh((u - u_i) / h)^2

    This is simpler and faster than traditional B-splines.
    """

    def __init__(
        self,
        grid_size: int = 5,
        grid_range: tuple = (-1.0, 1.0),
        denominator: float = 2.0,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.grid_range = grid_range

        # Create grid points
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        self.register_buffer("grid", grid)

        # Inverse denominator (1/h)
        self.inv_denominator = denominator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RSWAF basis values.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Basis values of shape (..., in_features, grid_size + 1)
        """
        # Expand dimensions for broadcasting
        # x: (..., in_features) -> (..., in_features, 1)
        # grid: (grid_size + 1,) -> (1, 1, grid_size + 1)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(*([1] * (x.dim())), -1)

        # Compute RSWAF: 1 - tanh((x - grid) / h)^2
        diff = (x_expanded - grid_expanded) * self.inv_denominator
        basis = 1.0 - torch.tanh(diff) ** 2

        return basis


class PureKANLinear(nn.Module):
    """Pure PyTorch implementation of a KAN linear layer.

    Uses RSWAF basis functions for simplicity and speed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,  # Not used in RSWAF, kept for API compatibility
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
        base_activation: str = "silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # Base activation
        if base_activation == "silu":
            self.base_activation = nn.SiLU()
        elif base_activation == "relu":
            self.base_activation = nn.ReLU()
        elif base_activation == "gelu":
            self.base_activation = nn.GELU()
        else:
            self.base_activation = nn.SiLU()

        # RSWAF basis
        self.basis = RSWAFBasis(grid_size, grid_range)

        # Base weight (for the activation function path)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * scale_base
        )

        # Spline weight (for the basis function path)
        # Shape: (out_features, in_features, grid_size + 1)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + 1)
            * scale_noise
            * scale_spline
        )

        # Layer normalization for input scaling
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        batch_size = x.shape[0]

        # Normalize input
        x_norm = self.layer_norm(x)

        # Base path: activation followed by linear
        if self.scale_base != 0:
            base_output = F.linear(self.base_activation(x_norm), self.base_weight)
        else:
            base_output = 0

        # Spline path: basis functions followed by weighted sum
        if self.scale_spline != 0:
            # Compute basis: (batch, in_features, grid_size + 1)
            basis = self.basis(x_norm)

            # Weighted sum: einsum for (b, i, g) * (o, i, g) -> (b, o)
            spline_output = torch.einsum("big,oig->bo", basis, self.spline_weight)
        else:
            spline_output = 0

        output = base_output + spline_output

        # Reshape back to original batch shape
        return output.view(*original_shape[:-1], self.out_features)

    def regularization_loss(self, p: int = 2) -> torch.Tensor:
        """Compute regularization loss on spline weights.

        Args:
            p: Norm order (default: 2 for L2)

        Returns:
            Regularization loss value.
        """
        return torch.norm(self.spline_weight, p=p)


class KANLayer(nn.Module):
    """Unified KAN layer interface.

    Wraps different KAN implementations (faster-kan, efficient-kan, or pure PyTorch)
    with a consistent interface.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
        base_activation: str = "silu",
        backend: Literal["faster", "efficient", "pure"] = "faster",
    ):
        super().__init__()
        self.backend = backend

        if backend == "faster" and _FASTER_KAN_AVAILABLE:
            self.layer = _FasterKANLayer(
                in_features,
                out_features,
                grid_min=grid_range[0],
                grid_max=grid_range[1],
                num_grids=grid_size,
            )
        elif backend == "efficient" and _EFFICIENT_KAN_AVAILABLE:
            self.layer = _EfficientKANLinear(
                in_features=in_features,
                out_features=out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                grid_range=list(grid_range),
                base_activation=nn.SiLU if base_activation == "silu" else nn.ReLU,
            )
        else:
            # Fall back to pure PyTorch implementation
            if backend == "faster" and not _FASTER_KAN_AVAILABLE:
                import warnings

                warnings.warn(
                    "faster-kan not available, using pure PyTorch implementation"
                )
            if backend == "efficient" and not _EFFICIENT_KAN_AVAILABLE:
                import warnings

                warnings.warn(
                    "efficient-kan not available, using pure PyTorch implementation"
                )

            self.layer = PureKANLinear(
                in_features=in_features,
                out_features=out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                grid_range=grid_range,
                base_activation=base_activation,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the KAN layer."""
        return self.layer(x)

    def regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from the layer."""
        if hasattr(self.layer, "regularization_loss"):
            return self.layer.regularization_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)


class KAN(nn.Module):
    """Multi-layer KAN network.

    Stacks multiple KAN layers to form a deep network.
    """

    def __init__(
        self,
        config: KANConfig,
        backend: Literal["faster", "efficient", "pure"] = "faster",
    ):
        super().__init__()
        self.config = config
        self.backend = backend

        # Build layers
        dims = config.get_layer_dims()
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(
                KANLayer(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    grid_size=config.grid_size,
                    spline_order=config.spline_order,
                    scale_noise=config.scale_noise,
                    scale_base=config.scale_base,
                    scale_spline=config.scale_spline,
                    grid_range=config.grid_range,
                    base_activation=config.base_activation,
                    backend=backend,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Output tensor of shape (..., output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self) -> torch.Tensor:
        """Sum of regularization losses from all layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.regularization_loss()
        return total


def get_kan_availability() -> dict:
    """Check which KAN backends are available.

    Returns:
        Dictionary with backend availability status.
    """
    return {
        "faster": _FASTER_KAN_AVAILABLE,
        "efficient": _EFFICIENT_KAN_AVAILABLE,
        "pure": True,  # Always available
    }
