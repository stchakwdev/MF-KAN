"""Error metrics for MFKAN evaluation.

This module implements metrics used in the MFKAN paper for
comparing model predictions against ground truth.
"""

from typing import Dict

import torch


def relative_l2_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute relative L2 error.

    From the paper:
        error = ||y_pred - y_true||_2 / ||y_true||_2

    Args:
        y_pred: Predicted values.
        y_true: True values.
        eps: Small value to avoid division by zero.

    Returns:
        Relative L2 error (scalar tensor).
    """
    diff_norm = torch.norm(y_pred - y_true, p=2)
    true_norm = torch.norm(y_true, p=2)
    return diff_norm / (true_norm + eps)


def mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Compute mean squared error.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        MSE (scalar tensor).
    """
    return torch.mean((y_pred - y_true) ** 2)


def rmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Compute root mean squared error.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        RMSE (scalar tensor).
    """
    return torch.sqrt(mse(y_pred, y_true))


def mae(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Compute mean absolute error.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        MAE (scalar tensor).
    """
    return torch.mean(torch.abs(y_pred - y_true))


def max_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Compute maximum absolute error.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        Max error (scalar tensor).
    """
    return torch.max(torch.abs(y_pred - y_true))


def r2_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Compute R-squared (coefficient of determination).

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        R^2 score (scalar tensor).
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_all_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """Compute all metrics.

    Args:
        y_pred: Predicted values.
        y_true: True values.

    Returns:
        Dictionary of all metrics.
    """
    return {
        "relative_l2": relative_l2_error(y_pred, y_true).item(),
        "mse": mse(y_pred, y_true).item(),
        "rmse": rmse(y_pred, y_true).item(),
        "mae": mae(y_pred, y_true).item(),
        "max_error": max_error(y_pred, y_true).item(),
        "r2": r2_score(y_pred, y_true).item(),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty print metrics.

    Args:
        metrics: Dictionary of metric values.
        prefix: Optional prefix for each line.
    """
    print(f"{prefix}Relative L2 Error: {metrics['relative_l2']:.6f}")
    print(f"{prefix}MSE: {metrics['mse']:.6f}")
    print(f"{prefix}RMSE: {metrics['rmse']:.6f}")
    print(f"{prefix}MAE: {metrics['mae']:.6f}")
    print(f"{prefix}Max Error: {metrics['max_error']:.6f}")
    print(f"{prefix}R^2 Score: {metrics['r2']:.6f}")
