"""Test 3: 2D nonlinear function.

From the paper (Eq. 13-14):
    f_L(x1, x2) = 0.5*(6*x2-2)^2*sin(12*x1-4) + 10*(x1-0.5) - 5
    f_H(x1, x2) = (6*x2-2)^2*sin(12*x1-4)

The HF function is approximately 2x the LF function plus a
nonlinear correction term.

Paper results: N_LF=10000, N_HF=150 -> relative L2 error ~0.02
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.utils import generate_test3_2d_nonlinear, compute_all_metrics


def run_test3(
    n_lf: int = 10000,
    n_hf: int = 150,
    lf_epochs: int = 500,
    hf_epochs: int = 500,
    plot: bool = True,
):
    """Run Test 3: 2D nonlinear function.

    Args:
        n_lf: Number of low-fidelity samples.
        n_hf: Number of high-fidelity samples.
        lf_epochs: Epochs for LF training.
        hf_epochs: Epochs for HF training.
        plot: Whether to show plot.

    Returns:
        Dictionary with results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate data
    data = generate_test3_2d_nonlinear(n_lf=n_lf, n_hf=n_hf, device=device)
    print(f"Data: {data.metadata['description']}")
    print(f"LF samples: {n_lf}, HF samples: {n_hf}")

    # Model configuration (2D input)
    model_config = MFKANConfig(
        input_dim=2,
        output_dim=1,
        lf_hidden_dims=[64, 64],
        nl_hidden_dims=[32, 32],
        grid_size=8,
        alpha_init=0.5,
        alpha_trainable=True,
        lambda_alpha=0.01,
        backend="pure",
    )

    # Training configuration
    train_config = TrainingConfig(
        lf_epochs=lf_epochs,
        hf_epochs=hf_epochs,
        lf_lr=1e-3,
        hf_lr=1e-3,
        lf_batch_size=256,
        hf_batch_size=32,
        verbose=True,
        log_interval=100,
    )

    # Train model
    model = MFKAN(model_config)
    trainer = MFKANTrainer(model, train_config, device=device)

    print("\n--- Phase 1: LF Training ---")
    trainer.train_low_fidelity(data.x_lf, data.y_lf)

    print("\n--- Phase 2: HF Training ---")
    trainer.train_high_fidelity(data.x_hf, data.y_hf)

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(data.x_test)

    metrics = compute_all_metrics(y_pred, data.y_test)

    print("\n--- Results ---")
    print(f"Relative L2 Error: {metrics['relative_l2']:.4f}")
    print(f"R^2 Score: {metrics['r2']:.4f}")
    print(f"Final alpha: {model.alpha.item():.4f}")

    # Plot
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Reshape for surface plot
        n_grid = int(np.sqrt(data.x_test.shape[0]))
        x1 = data.x_test[:, 0].cpu().numpy().reshape(n_grid, n_grid)
        x2 = data.x_test[:, 1].cpu().numpy().reshape(n_grid, n_grid)
        y_true = data.y_test.cpu().numpy().reshape(n_grid, n_grid)
        y_p = y_pred.cpu().numpy().reshape(n_grid, n_grid)
        error = np.abs(y_true - y_p)

        # True function
        c1 = axes[0].contourf(x1, x2, y_true, levels=50, cmap="viridis")
        plt.colorbar(c1, ax=axes[0])
        axes[0].scatter(
            data.x_hf[:, 0].cpu().numpy(),
            data.x_hf[:, 1].cpu().numpy(),
            c="red",
            s=20,
            alpha=0.8,
            label=f"HF data (n={n_hf})",
        )
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("x2")
        axes[0].set_title("True HF Function")
        axes[0].legend()

        # Prediction
        c2 = axes[1].contourf(x1, x2, y_p, levels=50, cmap="viridis")
        plt.colorbar(c2, ax=axes[1])
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")
        axes[1].set_title(f"MFKAN Prediction (Error: {metrics['relative_l2']:.4f})")

        # Error
        c3 = axes[2].contourf(x1, x2, error, levels=50, cmap="Reds")
        plt.colorbar(c3, ax=axes[2])
        axes[2].set_xlabel("x1")
        axes[2].set_ylabel("x2")
        axes[2].set_title(f"Absolute Error (Max: {error.max():.4f})")

        plt.tight_layout()
        plt.savefig("test3_results.png", dpi=150)
        plt.show()

    return {
        "metrics": metrics,
        "alpha": model.alpha.item(),
        "model": model,
        "trainer": trainer,
    }


if __name__ == "__main__":
    results = run_test3()
