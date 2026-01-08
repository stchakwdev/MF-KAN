"""Test 1: Jump function with linear correlation.

From the paper (Eq. 9-10):
    f_L(x) = x                  for x in [-1, 1]
    f_H(x) = x + 1              for x <= 0
    f_H(x) = x - 1              for x > 0

This test demonstrates MFKAN's ability to learn a discontinuous
high-fidelity function from sparse data using linear correlation
with abundant low-fidelity data.

Paper results: N_LF=50, N_HF=5 -> relative L2 error ~0.06
"""

import torch
import matplotlib.pyplot as plt

from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.utils import generate_test1_jump_function, compute_all_metrics


def run_test1(
    n_lf: int = 50,
    n_hf: int = 5,
    lf_epochs: int = 500,
    hf_epochs: int = 500,
    plot: bool = True,
):
    """Run Test 1: Jump function.

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
    data = generate_test1_jump_function(n_lf=n_lf, n_hf=n_hf, device=device)
    print(f"Data: {data.metadata['description']}")
    print(f"LF samples: {n_lf}, HF samples: {n_hf}")

    # Model configuration
    model_config = MFKANConfig(
        input_dim=1,
        output_dim=1,
        lf_hidden_dims=[32, 32],
        nl_hidden_dims=[16, 16],
        grid_size=5,
        alpha_init=0.5,
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

    metrics = compute_all_metrics(y_pred, data.y_test)

    print("\n--- Results ---")
    print(f"Relative L2 Error: {metrics['relative_l2']:.4f}")
    print(f"R^2 Score: {metrics['r2']:.4f}")
    print(f"Final alpha: {model.alpha.item():.4f}")

    # Plot
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        x = data.x_test.cpu().numpy()
        y_true = data.y_test.cpu().numpy()
        y_p = y_pred.cpu().numpy()

        # Predictions
        axes[0].plot(x, y_true, "b-", label="True HF", linewidth=2)
        axes[0].plot(x, y_p, "r--", label="MFKAN", linewidth=2)
        axes[0].scatter(
            data.x_hf.cpu().numpy(),
            data.y_hf.cpu().numpy(),
            c="green",
            s=100,
            zorder=5,
            label=f"HF data (n={n_hf})",
        )
        axes[0].scatter(
            data.x_lf.cpu().numpy(),
            data.y_lf.cpu().numpy(),
            c="orange",
            s=20,
            alpha=0.5,
            label=f"LF data (n={n_lf})",
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Test 1: Jump Function (Error: {metrics['relative_l2']:.4f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Training loss
        lf_loss = [h["total"] for h in trainer.history["lf"]]
        hf_loss = [h["total"] for h in trainer.history["hf"]]
        axes[1].semilogy(lf_loss, "b-", label="LF Loss")
        axes[1].semilogy(
            range(len(lf_loss), len(lf_loss) + len(hf_loss)),
            hf_loss,
            "r-",
            label="HF Loss",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training History")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("test1_results.png", dpi=150)
        plt.show()

    return {
        "metrics": metrics,
        "alpha": model.alpha.item(),
        "model": model,
        "trainer": trainer,
    }


if __name__ == "__main__":
    results = run_test1()
