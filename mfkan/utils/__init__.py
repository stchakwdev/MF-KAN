"""MFKAN utilities."""

from mfkan.utils.data import (
    MultifidelityData,
    generate_test1_jump_function,
    generate_test2_nonlinear_correlation,
    generate_test3_2d_nonlinear,
    generate_test4_higher_dimensional,
    generate_test5_poisson,
    generate_test6_mechanical_mnist,
    generate_test7_extrapolation,
    get_test_data,
    list_tests,
)
from mfkan.utils.metrics import (
    relative_l2_error,
    mse,
    rmse,
    mae,
    max_error,
    r2_score,
    compute_all_metrics,
    print_metrics,
)

__all__ = [
    # Data generators
    "MultifidelityData",
    "generate_test1_jump_function",
    "generate_test2_nonlinear_correlation",
    "generate_test3_2d_nonlinear",
    "generate_test4_higher_dimensional",
    "generate_test5_poisson",
    "generate_test6_mechanical_mnist",
    "generate_test7_extrapolation",
    "get_test_data",
    "list_tests",
    # Metrics
    "relative_l2_error",
    "mse",
    "rmse",
    "mae",
    "max_error",
    "r2_score",
    "compute_all_metrics",
    "print_metrics",
]
