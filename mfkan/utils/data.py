"""Test data generators for MFKAN.

This module implements all 7 test cases from the MFKAN paper:
- Test 1: Jump function with linear correlation
- Test 2: Nonlinear correlation (sin^2)
- Test 3: 2D nonlinear function
- Test 4: Higher-dimensional (4D) with noise
- Test 5: Physics-informed (Poisson equation)
- Test 6: Mechanical MNIST (placeholder)
- Test 7: Multifidelity extrapolation

Reference:
    Howard et al. (2024) "Multifidelity Kolmogorov-Arnold Networks"
    arXiv:2410.14764
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class MultifidelityData:
    """Container for multifidelity training data.

    Attributes:
        x_lf: Low-fidelity input data.
        y_lf: Low-fidelity target data.
        x_hf: High-fidelity input data.
        y_hf: High-fidelity target data.
        x_test: Test input data.
        y_test: True high-fidelity test values.
        metadata: Additional information about the test case.
    """

    x_lf: torch.Tensor
    y_lf: torch.Tensor
    x_hf: torch.Tensor
    y_hf: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    metadata: Dict[str, any]


def generate_test1_jump_function(
    n_lf: int = 50,
    n_hf: int = 5,
    n_test: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 1: Jump function with linear correlation.

    From the paper (Eq. 9-10):
        f_L(x) = x                              for x in [-1, 1]
        f_H(x) = x + 1                          for x <= 0
        f_H(x) = x - 1                          for x > 0

    The LF and HF functions are linearly correlated (same slope),
    but with a jump discontinuity at x=0.

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        n_test: Number of test samples.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing all datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Low-fidelity function: f_L(x) = x
    def f_lf(x):
        return x

    # High-fidelity function: f_H(x) with jump
    def f_hf(x):
        return torch.where(x <= 0, x + 1, x - 1)

    # Generate samples
    x_lf = torch.linspace(-1, 1, n_lf, device=device).unsqueeze(-1)
    y_lf = f_lf(x_lf)

    x_hf = torch.linspace(-1, 1, n_hf, device=device).unsqueeze(-1)
    y_hf = f_hf(x_hf)

    x_test = torch.linspace(-1, 1, n_test, device=device).unsqueeze(-1)
    y_test = f_hf(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test1_jump_function",
            "description": "Jump function with linear correlation",
            "input_dim": 1,
            "output_dim": 1,
            "correlation": "linear",
            "paper_ref": "Eq. 9-10",
        },
    )


def generate_test2_nonlinear_correlation(
    n_lf: int = 51,
    n_hf: int = 14,
    n_test: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 2: Nonlinear correlation.

    From the paper (Eq. 11-12):
        f_L(x) = sin(2*pi*x)                    for x in [0, 1]
        f_H(x) = sin^2(2*pi*x)                  for x in [0, 1]

    The relationship is nonlinear: f_H = f_L^2 (when f_L >= 0).

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        n_test: Number of test samples.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing all datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Low-fidelity function
    def f_lf(x):
        return torch.sin(2 * np.pi * x)

    # High-fidelity function
    def f_hf(x):
        return torch.sin(2 * np.pi * x) ** 2

    # Generate samples
    x_lf = torch.linspace(0, 1, n_lf, device=device).unsqueeze(-1)
    y_lf = f_lf(x_lf)

    x_hf = torch.linspace(0, 1, n_hf, device=device).unsqueeze(-1)
    y_hf = f_hf(x_hf)

    x_test = torch.linspace(0, 1, n_test, device=device).unsqueeze(-1)
    y_test = f_hf(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test2_nonlinear_correlation",
            "description": "Nonlinear correlation (sin to sin^2)",
            "input_dim": 1,
            "output_dim": 1,
            "correlation": "nonlinear",
            "paper_ref": "Eq. 11-12",
        },
    )


def generate_test3_2d_nonlinear(
    n_lf: int = 10000,
    n_hf: int = 150,
    n_test: int = 2500,
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 3: 2D nonlinear function.

    From the paper (Eq. 13-14):
        f_L(x1, x2) = 0.5 * (6*x2 - 2)^2 * sin(12*x1 - 4) + 10*(x1 - 0.5) - 5
        f_H(x1, x2) = (6*x2 - 2)^2 * sin(12*x1 - 4)

    The HF function is roughly 2 times the LF function plus a nonlinear correction.

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        n_test: Number of test samples (per dimension, so n_test^2 total for grid).
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing all datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Low-fidelity function
    def f_lf(x):
        x1, x2 = x[..., 0], x[..., 1]
        term1 = 0.5 * (6 * x2 - 2) ** 2 * torch.sin(12 * x1 - 4)
        term2 = 10 * (x1 - 0.5) - 5
        return (term1 + term2).unsqueeze(-1)

    # High-fidelity function
    def f_hf(x):
        x1, x2 = x[..., 0], x[..., 1]
        return ((6 * x2 - 2) ** 2 * torch.sin(12 * x1 - 4)).unsqueeze(-1)

    # Generate LF samples (random)
    x_lf = torch.rand(n_lf, 2, device=device)
    y_lf = f_lf(x_lf)

    # Generate HF samples (random)
    x_hf = torch.rand(n_hf, 2, device=device)
    y_hf = f_hf(x_hf)

    # Generate test samples (grid)
    n_per_dim = int(np.sqrt(n_test))
    x1_test = torch.linspace(0, 1, n_per_dim, device=device)
    x2_test = torch.linspace(0, 1, n_per_dim, device=device)
    x1_grid, x2_grid = torch.meshgrid(x1_test, x2_test, indexing="ij")
    x_test = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=-1)
    y_test = f_hf(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test3_2d_nonlinear",
            "description": "2D nonlinear function",
            "input_dim": 2,
            "output_dim": 1,
            "correlation": "nonlinear",
            "paper_ref": "Eq. 13-14",
        },
    )


def generate_test4_higher_dimensional(
    n_lf: int = 25000,
    n_hf: int = 150,
    n_test: int = 5000,
    noise_std: float = 0.1,
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 4: Higher-dimensional (4D) problem with noise.

    From the paper (Eq. 15-16):
        f_L(x) = sum_{i=1}^{4} sin(x_i) + noise
        f_H(x) = sum_{i=1}^{4} sin(x_i)

    The LF data has additive Gaussian noise, and HF is the clean function.

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        n_test: Number of test samples.
        noise_std: Standard deviation of noise in LF data.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing all datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = 4

    # High-fidelity function (clean)
    def f_hf(x):
        return torch.sin(x).sum(dim=-1, keepdim=True)

    # Low-fidelity function (noisy)
    def f_lf(x, noise_std=noise_std):
        y = f_hf(x)
        noise = torch.randn_like(y) * noise_std
        return y + noise

    # Generate LF samples (random in [-pi, pi])
    x_lf = torch.rand(n_lf, input_dim, device=device) * 2 * np.pi - np.pi
    y_lf = f_lf(x_lf)

    # Generate HF samples (random, same domain)
    x_hf = torch.rand(n_hf, input_dim, device=device) * 2 * np.pi - np.pi
    y_hf = f_hf(x_hf)

    # Generate test samples
    x_test = torch.rand(n_test, input_dim, device=device) * 2 * np.pi - np.pi
    y_test = f_hf(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test4_higher_dimensional",
            "description": "4D function with noisy LF data",
            "input_dim": input_dim,
            "output_dim": 1,
            "correlation": "linear (with noise)",
            "noise_std": noise_std,
            "paper_ref": "Eq. 15-16",
        },
    )


def generate_test5_poisson(
    n_lf: int = 1000,
    n_hf: int = 100,
    n_test: int = 500,
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 5: Physics-informed (Poisson equation).

    1D Poisson equation: -u''(x) = f(x) on [0, 1]
    with boundary conditions u(0) = u(1) = 0.

    LF: Coarse finite difference solution
    HF: Fine finite difference solution (or analytical)

    For the test case:
        f(x) = sin(pi*x)
        u_exact(x) = sin(pi*x) / pi^2

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        n_test: Number of test samples.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing all datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Analytical solution (high-fidelity)
    def u_exact(x):
        return torch.sin(np.pi * x) / (np.pi ** 2)

    # Low-fidelity: Coarse approximation with perturbation
    def u_lf(x):
        # Simplified model with slight error
        return torch.sin(np.pi * x) / (np.pi ** 2) * 0.95 + 0.01 * x * (1 - x)

    # Generate samples
    x_lf = torch.linspace(0, 1, n_lf, device=device).unsqueeze(-1)
    y_lf = u_lf(x_lf)

    x_hf = torch.linspace(0, 1, n_hf, device=device).unsqueeze(-1)
    y_hf = u_exact(x_hf)

    x_test = torch.linspace(0, 1, n_test, device=device).unsqueeze(-1)
    y_test = u_exact(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test5_poisson",
            "description": "1D Poisson equation (physics-informed)",
            "input_dim": 1,
            "output_dim": 1,
            "correlation": "linear",
            "equation": "-u''(x) = sin(pi*x)",
            "paper_ref": "Test 5",
        },
    )


def generate_test6_mechanical_mnist(
    n_lf: int = 1000,
    n_hf: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 6: Mechanical MNIST (placeholder).

    Note: This is a placeholder. The actual Mechanical MNIST dataset
    requires downloading the dataset from:
    https://open.bu.edu/handle/2144/39371

    For now, we generate synthetic data that mimics the structure:
    - Input: Flattened image features (e.g., 784 for 28x28)
    - Output: Mechanical response (strain energy, displacement, etc.)

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing synthetic data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reduced input dimension for synthetic test
    input_dim = 64

    # Synthetic response function
    def response_hf(x):
        # Nonlinear combination of input features
        return (
            torch.sum(x ** 2, dim=-1, keepdim=True) * 0.1
            + torch.sum(torch.sin(x * 2), dim=-1, keepdim=True) * 0.05
        )

    def response_lf(x):
        # Linear approximation
        return torch.sum(x ** 2, dim=-1, keepdim=True) * 0.1

    # Generate samples
    x_lf = torch.randn(n_lf, input_dim, device=device)
    y_lf = response_lf(x_lf)

    x_hf = torch.randn(n_hf, input_dim, device=device)
    y_hf = response_hf(x_hf)

    n_test = 500
    x_test = torch.randn(n_test, input_dim, device=device)
    y_test = response_hf(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test6_mechanical_mnist",
            "description": "Mechanical MNIST (synthetic placeholder)",
            "input_dim": input_dim,
            "output_dim": 1,
            "note": "Placeholder - use actual dataset for paper reproduction",
            "dataset_url": "https://open.bu.edu/handle/2144/39371",
            "paper_ref": "Test 6",
        },
    )


def generate_test7_extrapolation(
    n_lf: int = 100,
    n_hf: int = 20,
    n_test: int = 200,
    lf_range: Tuple[float, float] = (0.0, 2.0),
    hf_range: Tuple[float, float] = (0.0, 1.0),
    test_range: Tuple[float, float] = (0.0, 2.0),
    seed: int = 42,
    device: str = "cpu",
) -> MultifidelityData:
    """Test 7: Multifidelity extrapolation.

    Tests MFKAN's ability to extrapolate beyond HF training data
    using the correlation learned from LF data.

    LF data covers a wider range than HF data.
    The test evaluates predictions in the extrapolation region.

    Function:
        f_L(x) = x^2 + 0.5*sin(2*pi*x)
        f_H(x) = x^2 + sin(2*pi*x)

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        n_test: Number of test samples.
        lf_range: Range for LF data [min, max].
        hf_range: Range for HF data [min, max] (subset of lf_range).
        test_range: Range for test data.
        seed: Random seed for reproducibility.
        device: Device for tensors.

    Returns:
        MultifidelityData containing all datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Low-fidelity function
    def f_lf(x):
        return x ** 2 + 0.5 * torch.sin(2 * np.pi * x)

    # High-fidelity function
    def f_hf(x):
        return x ** 2 + torch.sin(2 * np.pi * x)

    # Generate LF samples (wider range)
    x_lf = torch.linspace(lf_range[0], lf_range[1], n_lf, device=device).unsqueeze(-1)
    y_lf = f_lf(x_lf)

    # Generate HF samples (restricted range)
    x_hf = torch.linspace(hf_range[0], hf_range[1], n_hf, device=device).unsqueeze(-1)
    y_hf = f_hf(x_hf)

    # Generate test samples (full range including extrapolation)
    x_test = torch.linspace(test_range[0], test_range[1], n_test, device=device).unsqueeze(-1)
    y_test = f_hf(x_test)

    return MultifidelityData(
        x_lf=x_lf,
        y_lf=y_lf,
        x_hf=x_hf,
        y_hf=y_hf,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "name": "test7_extrapolation",
            "description": "Multifidelity extrapolation test",
            "input_dim": 1,
            "output_dim": 1,
            "lf_range": lf_range,
            "hf_range": hf_range,
            "test_range": test_range,
            "extrapolation_region": f"[{hf_range[1]}, {test_range[1]}]",
            "paper_ref": "Test 7",
        },
    )


# Registry of all test generators
TEST_GENERATORS = {
    "test1": generate_test1_jump_function,
    "test2": generate_test2_nonlinear_correlation,
    "test3": generate_test3_2d_nonlinear,
    "test4": generate_test4_higher_dimensional,
    "test5": generate_test5_poisson,
    "test6": generate_test6_mechanical_mnist,
    "test7": generate_test7_extrapolation,
}


def get_test_data(
    test_name: str,
    **kwargs,
) -> MultifidelityData:
    """Get test data by name.

    Args:
        test_name: Name of the test (test1-test7).
        **kwargs: Additional arguments passed to the generator.

    Returns:
        MultifidelityData for the specified test.

    Raises:
        ValueError: If test_name is not recognized.
    """
    if test_name not in TEST_GENERATORS:
        available = ", ".join(TEST_GENERATORS.keys())
        raise ValueError(f"Unknown test: {test_name}. Available: {available}")

    return TEST_GENERATORS[test_name](**kwargs)


def list_tests() -> Dict[str, str]:
    """List all available test cases.

    Returns:
        Dictionary mapping test names to descriptions.
    """
    descriptions = {
        "test1": "Jump function with linear correlation (Eq. 9-10)",
        "test2": "Nonlinear correlation sin to sin^2 (Eq. 11-12)",
        "test3": "2D nonlinear function (Eq. 13-14)",
        "test4": "4D function with noisy LF data (Eq. 15-16)",
        "test5": "1D Poisson equation (physics-informed)",
        "test6": "Mechanical MNIST (synthetic placeholder)",
        "test7": "Multifidelity extrapolation",
    }
    return descriptions
