"""Multifidelity Kolmogorov-Arnold Networks (MFKAN).

A PyTorch implementation of MFKANs for training accurate models using
low-fidelity data combined with sparse high-fidelity data.

Based on: Howard et al. (2024) "Multifidelity Kolmogorov-Arnold Networks"
arXiv:2410.14764
"""

from mfkan.version import __version__
from mfkan.configs.config import KANConfig, MFKANConfig, TrainingConfig
from mfkan.models.mfkan import MFKAN
from mfkan.models.linear_kan import LinearKAN
from mfkan.models.kan_backend import KANLayer, KAN
from mfkan.training.loss import MFKANLoss
from mfkan.training.trainer import MFKANTrainer

__all__ = [
    "__version__",
    # Configs
    "KANConfig",
    "MFKANConfig",
    "TrainingConfig",
    # Models
    "MFKAN",
    "LinearKAN",
    "KANLayer",
    "KAN",
    # Training
    "MFKANLoss",
    "MFKANTrainer",
]
