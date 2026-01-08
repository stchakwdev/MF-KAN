"""MFKAN trainer for two-phase training.

This module implements the two-phase training procedure for MFKAN:
1. Phase 1: Train K_L (low-fidelity KAN) on low-fidelity data
2. Phase 2: Freeze K_L, train K_l, K_nl, and alpha on high-fidelity data
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mfkan.configs.config import TrainingConfig
from mfkan.models.mfkan import MFKAN
from mfkan.training.loss import MFKANLoss, LFKANLoss


class MFKANTrainer:
    """Two-phase trainer for MFKAN.

    Training procedure:
    1. train_low_fidelity(): Train K_L on abundant low-fidelity data
    2. train_high_fidelity(): Freeze K_L, train correction networks on HF data

    Args:
        model: MFKAN model to train.
        config: Training configuration.
        device: Device to use (overrides config if provided).

    Example:
        >>> model = MFKAN(model_config)
        >>> trainer = MFKANTrainer(model, train_config)
        >>> trainer.train_low_fidelity(x_lf, y_lf)
        >>> trainer.train_high_fidelity(x_hf, y_hf)
        >>> y_pred = model(x_test)
    """

    def __init__(
        self,
        model: MFKAN,
        config: TrainingConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or config.get_device()
        self.model.to(self.device)

        # Training history
        self.history: Dict[str, List[Dict[str, float]]] = {
            "lf": [],
            "hf": [],
        }

        # Set random seed
        torch.manual_seed(config.seed)

    def train_low_fidelity(
        self,
        x_lf: torch.Tensor,
        y_lf: torch.Tensor,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> Dict[str, List[float]]:
        """Phase 1: Train low-fidelity model K_L.

        Args:
            x_lf: Low-fidelity input data of shape (N_LF, input_dim).
            y_lf: Low-fidelity target data of shape (N_LF, output_dim).
            val_data: Optional validation data tuple (x_val, y_val).
            callback: Optional callback function called each epoch.

        Returns:
            Training history dictionary.
        """
        self.model.train()

        # Setup optimizer for K_L only
        optimizer = torch.optim.Adam(
            self.model.get_trainable_lf_params(),
            lr=self.config.lf_lr,
            weight_decay=self.config.weight_decay,
        )

        # Setup data loader
        dataset = TensorDataset(
            x_lf.to(self.device),
            y_lf.to(self.device),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.lf_batch_size,
            shuffle=True,
        )

        # Loss function
        loss_fn = LFKANLoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.lf_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad()

                # Forward pass
                y_pred = self.model.forward_lf(x_batch)

                # Compute loss
                loss, components = loss_fn(y_pred, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                for k, v in components.items():
                    epoch_metrics[k] += v
                num_batches += 1

            # Average metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= num_batches

            # Validation
            if val_data is not None:
                val_loss = self._validate_lf(val_data)
                epoch_metrics["val_loss"] = val_loss

                # Early stopping
                if self.config.early_stopping_patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stopping_patience:
                            if self.config.verbose:
                                print(f"Early stopping at epoch {epoch}")
                            break

            # Store history
            self.history["lf"].append(dict(epoch_metrics))

            # Logging
            if self.config.verbose and (epoch + 1) % self.config.log_interval == 0:
                loss_str = f"Epoch {epoch + 1}/{self.config.lf_epochs} - "
                loss_str += f"LF Loss: {epoch_metrics['total']:.6f}"
                if val_data is not None:
                    loss_str += f" - Val Loss: {epoch_metrics['val_loss']:.6f}"
                print(loss_str)

            # Callback
            if callback is not None:
                callback(epoch, dict(epoch_metrics))

        # Mark LF as trained and freeze
        self.model.mark_lf_trained()
        self.model.freeze_low_fidelity()

        if self.config.verbose:
            print("Low-fidelity training complete. K_L is now frozen.")

        return {"lf": [h["total"] for h in self.history["lf"]]}

    def train_high_fidelity(
        self,
        x_hf: torch.Tensor,
        y_hf: torch.Tensor,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> Dict[str, List[float]]:
        """Phase 2: Train high-fidelity correction networks.

        Trains K_l, K_nl, and alpha while K_L is frozen.

        Args:
            x_hf: High-fidelity input data of shape (N_HF, input_dim).
            y_hf: High-fidelity target data of shape (N_HF, output_dim).
            val_data: Optional validation data tuple (x_val, y_val).
            callback: Optional callback function called each epoch.

        Returns:
            Training history dictionary.

        Raises:
            RuntimeError: If low-fidelity model hasn't been trained.
        """
        if not self.model.is_lf_trained:
            raise RuntimeError(
                "Low-fidelity model must be trained first. "
                "Call train_low_fidelity() before train_high_fidelity()."
            )

        self.model.train()

        # Setup optimizer for HF components only
        optimizer = torch.optim.Adam(
            self.model.get_trainable_hf_params(),
            lr=self.config.hf_lr,
            weight_decay=self.config.weight_decay,
        )

        # Setup data loader
        dataset = TensorDataset(
            x_hf.to(self.device),
            y_hf.to(self.device),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.hf_batch_size,
            shuffle=True,
        )

        # Loss function
        loss_fn = MFKANLoss(
            lambda_alpha=self.model.config.lambda_alpha,
            alpha_exponent=self.model.config.alpha_exponent,
            regularization_weight=self.model.config.regularization_weight,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.hf_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad()

                # Forward pass
                y_pred = self.model.forward_hf(x_batch)

                # Compute loss
                loss, components = loss_fn(
                    y_pred,
                    y_batch,
                    self.model.alpha,
                    self.model.regularization_loss(),
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                for k, v in components.items():
                    epoch_metrics[k] += v
                num_batches += 1

            # Average metrics
            for k in epoch_metrics:
                epoch_metrics[k] /= num_batches

            # Validation
            if val_data is not None:
                val_loss = self._validate_hf(val_data)
                epoch_metrics["val_loss"] = val_loss

                # Early stopping
                if self.config.early_stopping_patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.early_stopping_patience:
                            if self.config.verbose:
                                print(f"Early stopping at epoch {epoch}")
                            break

            # Store history
            self.history["hf"].append(dict(epoch_metrics))

            # Logging
            if self.config.verbose and (epoch + 1) % self.config.log_interval == 0:
                loss_str = f"Epoch {epoch + 1}/{self.config.hf_epochs} - "
                loss_str += f"HF Loss: {epoch_metrics['total']:.6f} - "
                loss_str += f"Alpha: {epoch_metrics['alpha']:.4f}"
                if val_data is not None:
                    loss_str += f" - Val Loss: {epoch_metrics['val_loss']:.6f}"
                print(loss_str)

            # Callback
            if callback is not None:
                callback(epoch, dict(epoch_metrics))

        if self.config.verbose:
            print(f"High-fidelity training complete. Final alpha: {self.model.alpha.item():.4f}")

        return {
            "hf": [h["total"] for h in self.history["hf"]],
            "alpha": [h["alpha"] for h in self.history["hf"]],
        }

    def _validate_lf(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """Compute validation loss for low-fidelity model."""
        self.model.eval()
        x_val, y_val = val_data
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)

        with torch.no_grad():
            y_pred = self.model.forward_lf(x_val)
            loss = nn.functional.mse_loss(y_pred, y_val)

        self.model.train()
        return loss.item()

    def _validate_hf(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> float:
        """Compute validation loss for high-fidelity model."""
        self.model.eval()
        x_val, y_val = val_data
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)

        with torch.no_grad():
            y_pred = self.model.forward_hf(x_val)
            loss = nn.functional.mse_loss(y_pred, y_val)

        self.model.train()
        return loss.item()

    def fit(
        self,
        x_lf: torch.Tensor,
        y_lf: torch.Tensor,
        x_hf: torch.Tensor,
        y_hf: torch.Tensor,
        val_lf: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        val_hf: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, List[float]]:
        """Convenience method for full two-phase training.

        Args:
            x_lf: Low-fidelity inputs.
            y_lf: Low-fidelity targets.
            x_hf: High-fidelity inputs.
            y_hf: High-fidelity targets.
            val_lf: Optional validation data for LF phase.
            val_hf: Optional validation data for HF phase.

        Returns:
            Combined training history.
        """
        if self.config.verbose:
            print("=" * 50)
            print("Phase 1: Low-Fidelity Training")
            print("=" * 50)

        lf_history = self.train_low_fidelity(x_lf, y_lf, val_data=val_lf)

        if self.config.verbose:
            print("\n" + "=" * 50)
            print("Phase 2: High-Fidelity Training")
            print("=" * 50)

        hf_history = self.train_high_fidelity(x_hf, y_hf, val_data=val_hf)

        return {**lf_history, **hf_history}

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "config": {
                "model": self.model.config,
                "training": self.config,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {"lf": [], "hf": []})
