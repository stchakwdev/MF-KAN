"""Linear KAN (K_l) for multifidelity correction.

The Linear KAN learns linear correlations between low-fidelity and high-fidelity data.
From the paper (Section 2.2.3):
- k=1: Linear (first-degree) polynomial splines
- wb=0: No base activation contribution
- ws=1: Unit spline scale
- g=1: Single grid interval (two grid points)
- No hidden layers: Direct input-to-output mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearKAN(nn.Module):
    """Linear KAN for learning linear correlations.

    This is a simplified KAN that only learns linear correlations,
    which is useful when the relationship between low-fidelity and
    high-fidelity data is predominantly linear.

    The linear correlation is learned as:
        y = W_s @ x + b

    where W_s are the spline weights learned over a minimal grid.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output features.
        grid_range: Range for grid points (default: (-1.0, 1.0)).
        bias: Whether to include bias (default: True).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_range = grid_range

        # Linear spline weights
        # With k=1 and g=1, we essentially have a linear transformation
        # We implement this efficiently as a standard linear layer
        # but with the conceptual understanding that it represents
        # linear B-spline basis functions
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("bias", None)

        # Optional: Layer norm for input normalization
        # This helps with grid alignment as noted in the paper
        self.use_layer_norm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Output tensor of shape (..., output_dim)
        """
        return F.linear(x, self.weight, self.bias)

    def regularization_loss(self) -> torch.Tensor:
        """Regularization loss (L2 norm of weights).

        Returns:
            L2 regularization loss.
        """
        return torch.norm(self.weight, p=2)


class LinearKANExtended(nn.Module):
    """Extended Linear KAN with piecewise linear capabilities.

    This version supports learning different linear correlations
    in different parts of the domain by using multiple grid points.
    From the paper (Section 2.2.3):
        "If it is known that the low-fidelity data and the high-fidelity
        data are linearly correlated, but with different correlations in
        different parts of the domain, we could increase the number of
        grid points to learn additional linear correlations."

    Args:
        input_dim: Number of input features.
        output_dim: Number of output features.
        num_segments: Number of linear segments (grid intervals).
        grid_range: Range for grid points.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_segments: int = 1,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_segments = num_segments
        self.grid_range = grid_range

        # Grid points
        grid = torch.linspace(grid_range[0], grid_range[1], num_segments + 1)
        self.register_buffer("grid", grid)

        if num_segments == 1:
            # Simple linear case
            self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            # Piecewise linear: weights for each segment
            # Shape: (num_segments + 1, output_dim, input_dim)
            # We use linear interpolation between grid points
            self.coeff = nn.Parameter(
                torch.randn(num_segments + 1, output_dim, input_dim) * 0.1
            )
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Output tensor of shape (..., output_dim)
        """
        if self.num_segments == 1:
            return F.linear(x, self.weight, self.bias)

        # Piecewise linear interpolation
        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.input_dim)
        batch_size = x_flat.shape[0]

        # Compute mean input for segment selection (simplified approach)
        x_mean = x_flat.mean(dim=-1, keepdim=True)

        # Find which segment each point belongs to
        # Compute basis weights using linear interpolation
        grid_expanded = self.grid.view(1, -1)
        diff = x_mean - grid_expanded  # (batch, num_segments + 1)

        # Distance to grid points (for soft assignment)
        h = (self.grid_range[1] - self.grid_range[0]) / self.num_segments
        weights = F.relu(1 - torch.abs(diff) / h)  # Triangle basis
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Normalize

        # Weighted combination of coefficients
        # coeff: (num_segments + 1, output_dim, input_dim)
        # weights: (batch, num_segments + 1)
        weighted_coeff = torch.einsum("bg,goi->boi", weights, self.coeff)

        # Apply weighted linear transformation
        output = torch.einsum("boi,bi->bo", weighted_coeff, x_flat) + self.bias

        return output.view(*batch_shape, self.output_dim)

    def regularization_loss(self) -> torch.Tensor:
        """Regularization loss."""
        if self.num_segments == 1:
            return torch.norm(self.weight, p=2)
        return torch.norm(self.coeff, p=2)
