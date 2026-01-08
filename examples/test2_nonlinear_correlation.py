"""Test 2: Nonlinear correlation.

From the paper (Eq. 11-12):
    f_L(x) = sin(2*pi*x)        for x in [0, 1]
    f_H(x) = sin^2(2*pi*x)      for x in [0, 1]

The relationship is nonlinear: f_H = f_L^2 (when f_L >= 0).
This test demonstrates MFKAN's ability to learn nonlinear
correlations between fidelity levels.

Paper results: N_LF=51, N_HF=14 -> captures the sin^2 pattern
"""

import torch
import matplotlib.pyplot as plt

from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.utils import generate_test2_nonlinear_correlation, compute_all_metrics


def run_test2(
    n_lf: int = 51,
    n_hf: int = 14,
    lf_epochs: int = 500,
    hf_epochs: int = 500,
    plot: bool = True,
):
    """Run Test 2: Nonlinear correlation.

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
    data = generate_test2_nonlinear_correlation(n_lf=n_lf, n_hf=n_hf, device=device)
    print(f"Data: {data.metadata['description']}")
    print(f"LF samples: {n_lf}, HF samples: {n_hf}")

    # Model configuration
    # For nonlinear correlation, we need the nonlinear KAN
    model_config = MFKANConfig(
        input_dim=1,
        output_dim=1,
        lf_hidden_dims=[32, 32],
        nl_hidden_dims=[32, 32],  # Larger for nonlinear learning
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
        y_lf_pred = model.forward_lf(data.x_test)

    metrics = compute_all_metrics(y_pred, data.y_test)

    print("\n--- Results ---")
    print(f"Relative L2 Error: {metrics['relative_l2']:.4f}")
    print(f"R^2 Score: {metrics['r2']:.4f}")
    print(f"Final alpha: {model.alpha.item():.4f}")
    print(f"(High alpha indicates nonlinear correlation is significant)")

    # Plot
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        x = data.x_test.cpu().numpy()
        y_true = data.y_test.cpu().numpy()
        y_p = y_pred.cpu().numpy()
        y_lf = y_lf_pred.cpu().numpy()

        # Predictions
        axes[0].plot(x, y_true, "b-", label="True HF (sin^2)", linewidth=2)
        axes[0].plot(x, y_p, "r--", label="MFKAN", linewidth=2)
        axes[0].plot(x, y_lf, "g:", label="LF model (sin)", linewidth=2, alpha=0.7)
        axes[0].scatter(
            data.x_hf.cpu().numpy(),
            data.y_hf.cpu().numpy(),
            c="red",
            s=60,
            zorder=5,
            label=f"HF data (n={n_hf})",
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Test 2: Nonlinear Correlation (Error: {metrics['relative_l2']:.4f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Correlation plot
        axes[1].scatter(y_lf.flatten(), y_true.flatten(), alpha=0.5, s=10, label="True")
        axes[1].scatter(y_lf.flatten(), y_p.flatten(), alpha=0.5, s=10, label="Predicted")
        axes[1].set_xlabel("LF output (sin)")
        axes[1].set_ylabel("HF output (sin^2)")
        axes[1].set_title("LF vs HF Correlation")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Training history with alpha
        hf_alpha = [h["alpha"] for h in trainer.history["hf"]]
        hf_loss = [h["total"] for h in trainer.history["hf"]]

        ax2 = axes[2]
        ax2.plot(hf_loss, "r-", label="HF Loss")
        ax2.set_xlabel("HF Epoch")
        ax2.set_ylabel("Loss", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        ax2b = ax2.twinx()
        ax2b.plot(hf_alpha, "b--", label="Alpha")
        ax2b.set_ylabel("Alpha", color="b")
        ax2b.tick_params(axis="y", labelcolor="b")
        ax2b.set_ylim(0, 1)

        ax2.set_title(f"HF Training (Final alpha: {model.alpha.item():.3f})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("test2_results.png", dpi=150)
        plt.show()

    return {
        "metrics": metrics,
        "alpha": model.alpha.item(),
        "model": model,
        "trainer": trainer,
    }


if __name__ == "__main__":
    results = run_test2()
