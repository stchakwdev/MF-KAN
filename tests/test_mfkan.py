"""Tests for MFKAN model."""

import pytest
import torch

from mfkan import MFKAN, MFKANConfig


class TestMFKANConfig:
    """Tests for MFKANConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MFKANConfig(input_dim=2, output_dim=1)
        assert config.input_dim == 2
        assert config.output_dim == 1
        assert config.backend == "faster"  # Default is faster-kan
        assert config.alpha_init == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = MFKANConfig(
            input_dim=4,
            output_dim=2,
            lf_hidden_dims=[64, 64],
            nl_hidden_dims=[32, 32],
            alpha_init=0.3,
            lambda_alpha=0.1,
        )
        assert config.input_dim == 4
        assert config.output_dim == 2
        assert config.lf_hidden_dims == [64, 64]
        assert config.alpha_init == 0.3
        assert config.lambda_alpha == 0.1

    def test_get_lf_config(self):
        """Test LF config extraction."""
        config = MFKANConfig(
            input_dim=2,
            output_dim=1,
            lf_hidden_dims=[32, 32],
        )
        lf_config = config.get_lf_config()
        assert lf_config.input_dim == 2
        assert lf_config.output_dim == 1
        assert lf_config.hidden_dims == [32, 32]

    def test_get_nl_config(self):
        """Test NL config extraction."""
        config = MFKANConfig(
            input_dim=2,
            output_dim=1,
            nl_hidden_dims=[16, 16],
        )
        nl_config = config.get_nl_config()
        # NL input is input_dim + output_dim (concatenated)
        assert nl_config.input_dim == 3
        assert nl_config.output_dim == 1
        assert nl_config.hidden_dims == [16, 16]


class TestMFKAN:
    """Tests for MFKAN model."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = MFKANConfig(
            input_dim=2,
            output_dim=1,
            lf_hidden_dims=[16, 16],
            nl_hidden_dims=[8, 8],
            backend="pure",
        )
        return MFKAN(config)

    def test_model_creation(self, model):
        """Test model creation."""
        assert model is not None
        assert not model.is_lf_trained
        assert not model.is_lf_frozen

    def test_alpha_property(self, model):
        """Test alpha clamping."""
        # Alpha should be clamped to [0, 1]
        model._alpha.data = torch.tensor(1.5)
        assert model.alpha.item() == 1.0

        model._alpha.data = torch.tensor(-0.5)
        assert model.alpha.item() == 0.0

        model._alpha.data = torch.tensor(0.5)
        assert model.alpha.item() == 0.5

    def test_forward_lf(self, model):
        """Test low-fidelity forward pass."""
        x = torch.randn(10, 2)
        y = model.forward_lf(x)
        assert y.shape == (10, 1)

    def test_forward_hf_before_lf_trained(self, model):
        """Test HF forward works even before LF trained."""
        x = torch.randn(10, 2)
        # Should work but LF not frozen
        y = model.forward_hf(x)
        assert y.shape == (10, 1)

    def test_forward_hf_with_components(self, model):
        """Test HF forward with component return."""
        x = torch.randn(10, 2)
        y, components = model.forward_hf(x, return_components=True)
        assert y.shape == (10, 1)
        assert "y_lf" in components
        assert "y_linear" in components
        assert "y_nonlinear" in components
        assert "alpha" in components

    def test_freeze_low_fidelity(self, model):
        """Test freezing LF parameters."""
        model.freeze_low_fidelity()
        assert model.is_lf_frozen

        for param in model.k_lf.parameters():
            assert not param.requires_grad

    def test_unfreeze_low_fidelity(self, model):
        """Test unfreezing LF parameters."""
        model.freeze_low_fidelity()
        model.unfreeze_low_fidelity()
        assert not model.is_lf_frozen

        for param in model.k_lf.parameters():
            assert param.requires_grad

    def test_mark_lf_trained(self, model):
        """Test marking LF as trained."""
        assert not model.is_lf_trained
        model.mark_lf_trained()
        assert model.is_lf_trained

    def test_get_trainable_params(self, model):
        """Test parameter getters."""
        lf_params = model.get_trainable_lf_params()
        hf_params = model.get_trainable_hf_params()

        assert len(lf_params) > 0
        assert len(hf_params) > 0

        # LF params should be from k_lf
        # HF params should include k_linear, k_nonlinear, and alpha

    def test_regularization_loss(self, model):
        """Test regularization loss computation."""
        reg_loss = model.regularization_loss()
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.ndim == 0  # Scalar

    def test_auto_mode_before_lf_trained(self, model):
        """Test auto mode returns LF when not trained."""
        x = torch.randn(10, 2)
        y_auto = model(x, mode="auto")
        y_lf = model.forward_lf(x)
        # Should be equal since LF not trained
        torch.testing.assert_close(y_auto, y_lf)

    def test_auto_mode_after_lf_trained(self, model):
        """Test auto mode returns HF when trained."""
        model.mark_lf_trained()
        x = torch.randn(10, 2)
        y_auto = model(x, mode="auto")
        y_hf = model.forward_hf(x)
        torch.testing.assert_close(y_auto, y_hf)

    def test_repr(self, model):
        """Test string representation."""
        repr_str = repr(model)
        assert "MFKAN" in repr_str
        assert "input_dim" in repr_str
        assert "alpha" in repr_str


class TestMFKANGradients:
    """Tests for MFKAN gradient flow."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = MFKANConfig(
            input_dim=1,
            output_dim=1,
            lf_hidden_dims=[8],
            nl_hidden_dims=[8],
            backend="pure",
        )
        return MFKAN(config)

    def test_lf_gradient_flow(self, model):
        """Test gradients flow through LF network."""
        x = torch.randn(5, 1, requires_grad=True)
        y = model.forward_lf(x)
        loss = y.sum()
        loss.backward()

        # Check K_L has gradients
        for param in model.k_lf.parameters():
            assert param.grad is not None

    def test_hf_gradient_flow_frozen(self, model):
        """Test gradients don't flow to frozen LF."""
        model.mark_lf_trained()
        model.freeze_low_fidelity()

        x = torch.randn(5, 1)
        y = model.forward_hf(x)
        loss = y.sum()
        loss.backward()

        # K_L should have no gradients (frozen)
        for param in model.k_lf.parameters():
            assert param.grad is None

        # K_linear and K_nonlinear should have gradients
        for param in model.k_linear.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_alpha_gradient(self, model):
        """Test alpha receives gradients."""
        model.mark_lf_trained()
        model.freeze_low_fidelity()

        x = torch.randn(5, 1)
        y = model.forward_hf(x)
        loss = y.sum()
        loss.backward()

        assert model._alpha.grad is not None
