"""Tests for MFKAN training components."""

import pytest
import torch

from mfkan import MFKAN, MFKANConfig, TrainingConfig, MFKANTrainer
from mfkan.training.loss import MFKANLoss, LFKANLoss


class TestLFKANLoss:
    """Tests for LFKANLoss."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = LFKANLoss()
        y_pred = torch.randn(10, 1)
        y_true = torch.randn(10, 1)

        loss, components = loss_fn(y_pred, y_true)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "mse" in components
        assert "total" in components

    def test_loss_zero_for_perfect_prediction(self):
        """Test loss is zero for perfect prediction."""
        loss_fn = LFKANLoss()
        y = torch.randn(10, 1)

        loss, _ = loss_fn(y, y)

        assert loss.item() < 1e-6


class TestMFKANLoss:
    """Tests for MFKANLoss."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = MFKANLoss(lambda_alpha=0.01, alpha_exponent=4)
        y_pred = torch.randn(10, 1)
        y_true = torch.randn(10, 1)
        alpha = torch.tensor(0.5)

        loss, components = loss_fn(y_pred, y_true, alpha)

        assert isinstance(loss, torch.Tensor)
        assert "mse" in components
        assert "alpha_penalty" in components
        assert "alpha" in components
        assert "total" in components

    def test_alpha_penalty(self):
        """Test alpha penalty increases with alpha."""
        loss_fn = MFKANLoss(lambda_alpha=0.1, alpha_exponent=2)
        y_pred = torch.zeros(10, 1)
        y_true = torch.zeros(10, 1)

        _, comp_low = loss_fn(y_pred, y_true, torch.tensor(0.1))
        _, comp_high = loss_fn(y_pred, y_true, torch.tensor(0.9))

        assert comp_high["alpha_penalty"] > comp_low["alpha_penalty"]

    def test_regularization(self):
        """Test regularization term."""
        loss_fn = MFKANLoss(regularization_weight=0.1)
        y_pred = torch.zeros(10, 1)
        y_true = torch.zeros(10, 1)
        alpha = torch.tensor(0.5)
        reg_loss = torch.tensor(1.0)

        _, components = loss_fn(y_pred, y_true, alpha, reg_loss)

        assert components["regularization"] == pytest.approx(0.1)


class TestMFKANTrainer:
    """Tests for MFKANTrainer."""

    @pytest.fixture
    def model_and_trainer(self):
        """Create model and trainer."""
        model_config = MFKANConfig(
            input_dim=1,
            output_dim=1,
            lf_hidden_dims=[8],
            nl_hidden_dims=[8],
            backend="pure",
        )
        train_config = TrainingConfig(
            lf_epochs=5,
            hf_epochs=5,
            lf_lr=1e-3,
            hf_lr=1e-3,
            verbose=False,
        )
        model = MFKAN(model_config)
        trainer = MFKANTrainer(model, train_config)
        return model, trainer

    def test_trainer_creation(self, model_and_trainer):
        """Test trainer creation."""
        model, trainer = model_and_trainer
        assert trainer.model is model
        assert trainer.device == "cpu"

    def test_train_low_fidelity(self, model_and_trainer):
        """Test LF training."""
        model, trainer = model_and_trainer

        x_lf = torch.randn(20, 1)
        y_lf = torch.randn(20, 1)

        history = trainer.train_low_fidelity(x_lf, y_lf)

        assert "lf" in history
        assert len(history["lf"]) > 0
        assert model.is_lf_trained
        assert model.is_lf_frozen

    def test_train_high_fidelity_requires_lf(self, model_and_trainer):
        """Test HF training requires LF trained first."""
        model, trainer = model_and_trainer

        x_hf = torch.randn(10, 1)
        y_hf = torch.randn(10, 1)

        with pytest.raises(RuntimeError, match="Low-fidelity model must be trained"):
            trainer.train_high_fidelity(x_hf, y_hf)

    def test_train_high_fidelity(self, model_and_trainer):
        """Test HF training."""
        model, trainer = model_and_trainer

        x_lf = torch.randn(20, 1)
        y_lf = torch.randn(20, 1)
        x_hf = torch.randn(5, 1)
        y_hf = torch.randn(5, 1)

        trainer.train_low_fidelity(x_lf, y_lf)
        history = trainer.train_high_fidelity(x_hf, y_hf)

        assert "hf" in history
        assert "alpha" in history

    def test_fit_method(self, model_and_trainer):
        """Test convenience fit method."""
        model, trainer = model_and_trainer

        x_lf = torch.randn(20, 1)
        y_lf = torch.randn(20, 1)
        x_hf = torch.randn(5, 1)
        y_hf = torch.randn(5, 1)

        history = trainer.fit(x_lf, y_lf, x_hf, y_hf)

        assert "lf" in history
        assert "hf" in history

    def test_training_history(self, model_and_trainer):
        """Test training history is recorded."""
        model, trainer = model_and_trainer

        x_lf = torch.randn(20, 1)
        y_lf = torch.randn(20, 1)

        trainer.train_low_fidelity(x_lf, y_lf)

        assert len(trainer.history["lf"]) == 5  # lf_epochs
        assert "total" in trainer.history["lf"][0]

    def test_callback(self, model_and_trainer):
        """Test callback is called."""
        model, trainer = model_and_trainer

        x_lf = torch.randn(20, 1)
        y_lf = torch.randn(20, 1)

        callback_calls = []

        def callback(epoch, metrics):
            callback_calls.append((epoch, metrics))

        trainer.train_low_fidelity(x_lf, y_lf, callback=callback)

        assert len(callback_calls) == 5  # lf_epochs


class TestMFKANTrainerValidation:
    """Tests for MFKANTrainer with validation."""

    def test_validation_lf(self):
        """Test LF training with validation."""
        model_config = MFKANConfig(
            input_dim=1,
            output_dim=1,
            lf_hidden_dims=[8],
            backend="pure",
        )
        train_config = TrainingConfig(
            lf_epochs=5,
            hf_epochs=5,
            verbose=False,
        )
        model = MFKAN(model_config)
        trainer = MFKANTrainer(model, train_config)

        x_lf = torch.randn(20, 1)
        y_lf = torch.randn(20, 1)
        x_val = torch.randn(5, 1)
        y_val = torch.randn(5, 1)

        trainer.train_low_fidelity(x_lf, y_lf, val_data=(x_val, y_val))

        # Check validation loss was recorded
        assert "val_loss" in trainer.history["lf"][0]

    def test_early_stopping(self):
        """Test early stopping."""
        model_config = MFKANConfig(
            input_dim=1,
            output_dim=1,
            lf_hidden_dims=[8],
            backend="pure",
        )
        train_config = TrainingConfig(
            lf_epochs=100,  # Many epochs
            early_stopping_patience=2,  # Stop early
            verbose=False,
        )
        model = MFKAN(model_config)
        trainer = MFKANTrainer(model, train_config)

        # Create data where LF fits perfectly quickly
        x_lf = torch.linspace(-1, 1, 20).unsqueeze(-1)
        y_lf = x_lf  # Linear function
        x_val = torch.linspace(-1, 1, 5).unsqueeze(-1)
        y_val = x_val

        trainer.train_low_fidelity(x_lf, y_lf, val_data=(x_val, y_val))

        # Should stop before 100 epochs
        assert len(trainer.history["lf"]) < 100
