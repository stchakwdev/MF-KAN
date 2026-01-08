"""Quickstart example for MFKAN.

This example demonstrates the basic usage of MFKAN for
multifidelity learning with a simple 1D function.
"""

import torch
import matplotlib.pyplot as plt

from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.utils import get_test_data, compute_all_metrics, print_metrics


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate test data (Test 1: Jump function)
    print("\n" + "=" * 50)
    print("Generating Test 1 data: Jump function")
    print("=" * 50)
    data = get_test_data("test1", n_lf=50, n_hf=5, device=device)
    print(f"LF samples: {data.x_lf.shape[0]}")
    print(f"HF samples: {data.x_hf.shape[0]}")
    print(f"Test samples: {data.x_test.shape[0]}")

    # Create model configuration
    model_config = MFKANConfig(
        input_dim=1,
        output_dim=1,
        lf_hidden_dims=[32, 32],
        nl_hidden_dims=[16, 16],
        grid_size=5,
        spline_order=3,
        alpha_init=0.5,
        alpha_trainable=True,
        lambda_alpha=0.01,
        alpha_exponent=4,
        backend="pure",  # Use "faster" if faster-kan is installed
    )

    # Create training configuration
    train_config = TrainingConfig(
        lf_epochs=500,
        hf_epochs=500,
        lf_lr=1e-3,
        hf_lr=1e-3,
        lf_batch_size=32,
        hf_batch_size=5,
        verbose=True,
        log_interval=100,
    )

    # Create model and trainer
    model = MFKAN(model_config)
    trainer = MFKANTrainer(model, train_config, device=device)

    print("\n" + "=" * 50)
    print("Phase 1: Training Low-Fidelity Model")
    print("=" * 50)
    trainer.train_low_fidelity(data.x_lf, data.y_lf)

    print("\n" + "=" * 50)
    print("Phase 2: Training High-Fidelity Model")
    print("=" * 50)
    trainer.train_high_fidelity(data.x_hf, data.y_hf)

    # Evaluate
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    model.eval()
    with torch.no_grad():
        y_pred = model(data.x_test)

    metrics = compute_all_metrics(y_pred, data.y_test)
    print_metrics(metrics)
    print(f"\nFinal alpha: {model.alpha.item():.4f}")

    # Compare with HF-only baseline (for context)
    print("\n" + "=" * 50)
    print("Comparison: HF-only vs MFKAN")
    print("=" * 50)
    print(f"MFKAN Relative L2 Error: {metrics['relative_l2']:.4f}")
    print("(Paper reports ~0.06 for MFKAN vs ~0.31 for HF-only)")

    # Plot results
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Data and predictions
        x_np = data.x_test.cpu().numpy()
        y_true_np = data.y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        axes[0].plot(x_np, y_true_np, "b-", label="True HF", linewidth=2)
        axes[0].plot(x_np, y_pred_np, "r--", label="MFKAN", linewidth=2)
        axes[0].scatter(
            data.x_hf.cpu().numpy(),
            data.y_hf.cpu().numpy(),
            c="green",
            s=100,
            zorder=5,
            label=f"HF data (n={data.x_hf.shape[0]})",
        )
        axes[0].scatter(
            data.x_lf.cpu().numpy(),
            data.y_lf.cpu().numpy(),
            c="orange",
            s=30,
            alpha=0.5,
            label=f"LF data (n={data.x_lf.shape[0]})",
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("MFKAN Prediction")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Error
        error = (y_pred_np - y_true_np).flatten()
        axes[1].plot(x_np, error, "k-", linewidth=2)
        axes[1].axhline(y=0, color="r", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("Error")
        axes[1].set_title("Prediction Error")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Training history
        lf_loss = [h["total"] for h in trainer.history["lf"]]
        hf_loss = [h["total"] for h in trainer.history["hf"]]

        ax3a = axes[2]
        ax3a.semilogy(lf_loss, "b-", label="LF Loss")
        ax3a.semilogy(
            range(len(lf_loss), len(lf_loss) + len(hf_loss)),
            hf_loss,
            "r-",
            label="HF Loss",
        )
        ax3a.set_xlabel("Epoch")
        ax3a.set_ylabel("Loss")
        ax3a.set_title("Training History")
        ax3a.legend()
        ax3a.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("quickstart_results.png", dpi=150)
        print("\nPlot saved to: quickstart_results.png")
        plt.show()
    except Exception as e:
        print(f"\nCould not create plot: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
