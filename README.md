<div align="center">

# MF-KAN

**Multifidelity Kolmogorov-Arnold Networks for Data-Efficient Learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2410.14764-b31b1b.svg)](https://arxiv.org/abs/2410.14764)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Quick Start](#quick-start) | [Architecture](#architecture) | [Results](#benchmark-results) | [Examples](#examples) | [Citation](#citation)

</div>

---

## Overview

PyTorch implementation of **Multifidelity Kolmogorov-Arnold Networks (MFKANs)** based on [Howard et al. (2024)](https://arxiv.org/abs/2410.14764).

### The Problem

> *How can we train accurate models when high-fidelity data is expensive and scarce, but low-fidelity data is abundant?*

MFKAN solves this by learning correlations between cheap low-fidelity (LF) simulations and expensive high-fidelity (HF) experiments, achieving **5-15x better accuracy** than HF-only training with limited data.

### Key Innovation

| Approach | LF Data | HF Data | Accuracy |
|----------|---------|---------|----------|
| HF-Only Training | - | 5 samples | Poor (0.31 error) |
| **MFKAN** | 50 samples | 5 samples | **Excellent (0.06 error)** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MFKAN ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: Low-Fidelity Training (Abundant Data)                           │
│   ═══════════════════════════════════════════════                          │
│                                                                             │
│       Input x ──────► ┌─────────────┐ ──────► f_L(x)                       │
│                       │    K_L      │         (LF prediction)              │
│                       │  (LF KAN)   │                                      │
│                       └─────────────┘                                      │
│                              │                                              │
│                              ▼                                              │
│                         [FREEZE]                                            │
│                                                                             │
│   PHASE 2: High-Fidelity Training (Sparse Data)                            │
│   ═══════════════════════════════════════════════                          │
│                                                                             │
│                       ┌─────────────┐                                      │
│              ┌───────►│    K_l      │───────┐                              │
│              │        │ (Linear KAN)│       │                              │
│              │        │   k=1       │       │  (1-α)                       │
│   (x, f_L) ──┤        └─────────────┘       │                              │
│              │                              ▼                              │
│              │        ┌─────────────┐    ┌─────┐                           │
│              └───────►│   K_nl      │───►│  +  │───► K_H(x) = f_L + corr   │
│                       │(Nonlin KAN) │    └─────┘     (HF prediction)       │
│                       │   k>1       │       ▲                              │
│                       └─────────────┘       │  α                           │
│                                             │                              │
│                       ┌─────────────┐       │                              │
│                       │    α        │───────┘                              │
│                       │ (learnable) │                                      │
│                       │   [0, 1]    │                                      │
│                       └─────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Output: K_H(x) = f_L(x) + α·K_nl(x, f_L) + (1-α)·K_l(x, f_L)
```

### Three KAN Components

| Component | Description | Spline Order | Purpose |
|-----------|-------------|--------------|---------|
| **K_L** (Low-Fidelity) | Standard KAN | k=3 | Learn LF surrogate from abundant data |
| **K_l** (Linear) | Linear KAN | k=1 | Capture linear LF-HF correlation |
| **K_nl** (Nonlinear) | Standard KAN | k=3 | Learn residual nonlinear corrections |

### Loss Function

```
L_H = MSE(K_H(x), y_HF) + λ_α·α^n + w·||Φ_nl||
      ─────────────────   ───────   ──────────
       Prediction Error    Alpha     Spline
                          Penalty    Regularization
```

| Term | Purpose | Default |
|------|---------|---------|
| MSE | Accuracy on HF data | - |
| α^n penalty | Encourage linear correlation when sufficient | λ=0.01, n=4 |
| Regularization | Prevent overfitting in K_nl | w=0.0 |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/stchakwdev/MF-KAN.git
cd MF-KAN

# Install with pip
pip install -e .

# With faster-kan backend (recommended, 3.33x faster)
pip install -e ".[faster]"

# With all optional dependencies
pip install -e ".[all]"
```

### Basic Usage

```python
import torch
from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.utils import get_test_data

# Generate test data (or use your own)
data = get_test_data("test1", n_lf=50, n_hf=5)

# Configure model
config = MFKANConfig(
    input_dim=1,
    output_dim=1,
    lf_hidden_dims=[32, 32],
    nl_hidden_dims=[16, 16],
)

# Create and train
model = MFKAN(config)
trainer = MFKANTrainer(model, TrainingConfig(verbose=True))

# Phase 1: Train on abundant LF data
trainer.train_low_fidelity(data.x_lf, data.y_lf)

# Phase 2: Fine-tune on sparse HF data (LF frozen)
trainer.train_high_fidelity(data.x_hf, data.y_hf)

# Predict high-fidelity output
y_pred = model(data.x_test)
print(f"Learned α (nonlinear blend): {model.alpha.item():.3f}")
```

### Run Examples

```bash
# Quick validation
python examples/quickstart.py

# Paper test cases
python examples/test1_jump_function.py
python examples/test2_nonlinear_correlation.py
python examples/test3_2d_nonlinear.py
```

---

## Benchmark Results

### Test Cases from Paper

| Test | Description | Input Dim | N_LF | N_HF | Challenge |
|------|-------------|-----------|------|------|-----------|
| **Test 1** | Jump function | 1D | 50 | 5 | Discontinuity with linear correlation |
| **Test 2** | sin → sin² | 1D | 51 | 14 | Nonlinear correlation |
| **Test 3** | 2D surface | 2D | 10,000 | 150 | Spatial variation |
| **Test 4** | 4D with noise | 4D | 25,000 | 150 | High-dim + noisy LF |
| **Test 5** | Poisson PDE | 1D | 1,000 | 100 | Physics-informed |
| **Test 6** | Mechanical MNIST | 64D | 1,000 | 100 | Complex mechanics |
| **Test 7** | Extrapolation | 1D | 100 | 20 | Beyond HF training range |

### Performance Comparison

| Test | HF-Only Error | MFKAN Error | Improvement |
|------|---------------|-------------|-------------|
| Test 1 (Jump) | 0.31 | **0.06** | **5.2x better** |
| Test 3 (2D) | 0.96 | **0.02** | **48x better** |
| Test 4 (4D) | 0.89 | **0.08** | **11x better** |

### When to Use MFKAN

| Scenario | Recommendation |
|----------|----------------|
| Abundant HF data | Standard KAN/MLP sufficient |
| **Sparse HF + abundant LF** | **MFKAN recommended** |
| Linear LF-HF correlation | α → 0 (linear KAN dominates) |
| Nonlinear correlation | α → 1 (nonlinear KAN dominates) |
| Unknown correlation | MFKAN learns optimal α |

---

## Configuration

### Model Configuration

```python
from mfkan import MFKANConfig

config = MFKANConfig(
    # Architecture
    input_dim=2,                    # Input dimension
    output_dim=1,                   # Output dimension
    lf_hidden_dims=[64, 64],        # K_L hidden layers
    nl_hidden_dims=[32, 32],        # K_nl hidden layers

    # KAN parameters
    grid_size=5,                    # B-spline grid intervals
    spline_order=3,                 # B-spline degree (k)
    grid_range=(-1.0, 1.0),         # Input normalization range

    # Alpha blending
    alpha_init=0.5,                 # Initial α value
    alpha_trainable=True,           # Learn α during HF training

    # Loss parameters
    lambda_alpha=0.01,              # α penalty weight
    alpha_exponent=4,               # α penalty power (n)
    regularization_weight=0.0,      # Spline regularization

    # Backend: "faster" (default), "efficient", or "pure"
    backend="faster",
)
```

### Training Configuration

```python
from mfkan import TrainingConfig

config = TrainingConfig(
    # Epochs
    lf_epochs=500,                  # Low-fidelity training
    hf_epochs=500,                  # High-fidelity training

    # Learning rates
    lf_lr=1e-3,                     # LF learning rate
    hf_lr=1e-3,                     # HF learning rate

    # Batch sizes
    lf_batch_size=32,               # LF batch size
    hf_batch_size=16,               # HF batch size (often = N_HF)

    # Early stopping
    early_stopping_patience=0,       # 0 = disabled

    # Logging
    verbose=True,
    log_interval=100,
)
```

---

## KAN Backends

MFKAN supports multiple KAN implementations:

| Backend | Speed | Installation | Best For |
|---------|-------|--------------|----------|
| **`faster`** | **3.33x faster** | `pip install ".[faster]"` | Production (default) |
| `efficient` | Baseline | `pip install ".[efficient]"` | Compatibility |
| `pure` | Moderate | Built-in | No dependencies |

```python
# Use faster-kan (recommended)
config = MFKANConfig(..., backend="faster")

# Fallback to pure PyTorch
config = MFKANConfig(..., backend="pure")
```

---

## Project Structure

```
MF-KAN/
├── mfkan/
│   ├── __init__.py               # Package exports
│   ├── version.py                # Version info
│   ├── configs/
│   │   └── config.py             # KANConfig, MFKANConfig, TrainingConfig
│   ├── models/
│   │   ├── kan_backend.py        # Unified KAN interface (faster/efficient/pure)
│   │   ├── linear_kan.py         # K_l: Linear KAN (k=1 splines)
│   │   └── mfkan.py              # Main MFKAN model
│   ├── training/
│   │   ├── loss.py               # MFKANLoss with α penalty
│   │   └── trainer.py            # Two-phase MFKANTrainer
│   └── utils/
│       ├── data.py               # All 7 test data generators
│       └── metrics.py            # Error metrics (rel. L2, MSE, R²)
├── examples/
│   ├── quickstart.py             # Basic usage example
│   ├── test1_jump_function.py    # Paper Test 1
│   ├── test2_nonlinear_correlation.py  # Paper Test 2
│   └── test3_2d_nonlinear.py     # Paper Test 3
├── tests/                        # Unit tests (52 tests)
├── pyproject.toml                # Package configuration
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## API Reference

### MFKAN Model

```python
model = MFKAN(config)

# Forward passes
y_lf = model.forward_lf(x)          # Low-fidelity only
y_hf = model.forward_hf(x)          # High-fidelity (full model)
y = model(x)                        # Auto-select based on training state

# Properties
model.alpha                         # Current α value [0, 1]
model.is_lf_trained                 # LF phase completed?
model.is_lf_frozen                  # LF parameters frozen?

# Methods
model.freeze_low_fidelity()         # Freeze K_L after LF training
model.get_trainable_hf_params()     # Parameters for HF phase
model.regularization_loss()         # Spline regularization from K_nl
```

### MFKANTrainer

```python
trainer = MFKANTrainer(model, config)

# Two-phase training
trainer.train_low_fidelity(x_lf, y_lf, val_data=None)
trainer.train_high_fidelity(x_hf, y_hf, val_data=None)

# Convenience method (both phases)
trainer.fit(x_lf, y_lf, x_hf, y_hf)

# Checkpoints
trainer.save_checkpoint("model.pt")
trainer.load_checkpoint("model.pt")

# Access training history
trainer.history["lf"]               # LF training metrics
trainer.history["hf"]               # HF training metrics
```

### Metrics

```python
from mfkan.utils import compute_all_metrics, relative_l2_error

# All metrics at once
metrics = compute_all_metrics(y_pred, y_true)
# Returns: {relative_l2, mse, rmse, mae, max_error, r2}

# Individual metrics
error = relative_l2_error(y_pred, y_true)
```

---

## Examples

### Test 1: Jump Function (Linear Correlation)

```python
from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.utils import generate_test1_jump_function, compute_all_metrics

# Data: f_L(x) = x, f_H(x) = x±1 (jump at x=0)
data = generate_test1_jump_function(n_lf=50, n_hf=5)

# Train MFKAN
model = MFKAN(MFKANConfig(input_dim=1, output_dim=1))
trainer = MFKANTrainer(model, TrainingConfig())
trainer.fit(data.x_lf, data.y_lf, data.x_hf, data.y_hf)

# Evaluate
y_pred = model(data.x_test)
metrics = compute_all_metrics(y_pred, data.y_test)
print(f"Relative L2 Error: {metrics['relative_l2']:.4f}")  # ~0.06
print(f"Final α: {model.alpha.item():.4f}")  # ~0 (linear)
```

### Test 2: Nonlinear Correlation

```python
from mfkan.utils import generate_test2_nonlinear_correlation

# Data: f_L(x) = sin(2πx), f_H(x) = sin²(2πx)
data = generate_test2_nonlinear_correlation(n_lf=51, n_hf=14)

# Nonlinear correlation requires larger K_nl
config = MFKANConfig(
    input_dim=1, output_dim=1,
    nl_hidden_dims=[32, 32],  # Larger for nonlinear
)
model = MFKAN(config)
# ... train as above

print(f"Final α: {model.alpha.item():.4f}")  # ~1 (nonlinear)
```

---

## References

### Papers

- [Multifidelity Kolmogorov-Arnold Networks (Howard et al., 2024)](https://arxiv.org/abs/2410.14764) - Original MFKAN paper
- [KAN: Kolmogorov-Arnold Networks (Liu et al., 2024)](https://arxiv.org/abs/2404.19756) - Original KAN paper

### Resources

- [faster-kan](https://github.com/AthanasiosDelis/faster-kan) - Fast KAN with RSWAF basis (3.33x faster)
- [efficient-kan](https://github.com/Blealtan/efficient-kan) - Efficient B-spline KAN
- [awesome-kan](https://github.com/mintisan/awesome-kan) - Comprehensive KAN resources

---

## Citation

```bibtex
@article{howard2024multifidelity,
    title={Multifidelity Kolmogorov-Arnold Networks},
    author={Howard, Amanda A and Perego, Mauro and Karniadakis, George Em and Stinis, Panos},
    journal={arXiv preprint arXiv:2410.14764},
    year={2024}
}
```

```bibtex
@misc{mfkan_implementation_2025,
    title={MF-KAN: PyTorch Implementation of Multifidelity Kolmogorov-Arnold Networks},
    author={Samuel T. Chakwera},
    year={2025},
    url={https://github.com/stchakwdev/MF-KAN}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Back to Top](#mf-kan)**

Built with PyTorch | Inspired by [Howard et al. (2024)](https://arxiv.org/abs/2410.14764)

</div>
