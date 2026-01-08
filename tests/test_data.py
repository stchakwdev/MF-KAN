"""Tests for data generators."""

import pytest
import torch

from mfkan.utils import (
    MultifidelityData,
    generate_test1_jump_function,
    generate_test2_nonlinear_correlation,
    generate_test3_2d_nonlinear,
    generate_test4_higher_dimensional,
    generate_test5_poisson,
    generate_test7_extrapolation,
    get_test_data,
    list_tests,
)


class TestMultifidelityData:
    """Tests for MultifidelityData container."""

    def test_data_structure(self):
        """Test data container structure."""
        data = generate_test1_jump_function(n_lf=10, n_hf=5, n_test=20)

        assert isinstance(data, MultifidelityData)
        assert data.x_lf.shape[0] == 10
        assert data.x_hf.shape[0] == 5
        assert data.x_test.shape[0] == 20
        assert "name" in data.metadata


class TestTest1JumpFunction:
    """Tests for Test 1: Jump function."""

    def test_output_shape(self):
        """Test output shapes."""
        data = generate_test1_jump_function(n_lf=50, n_hf=5, n_test=100)

        assert data.x_lf.shape == (50, 1)
        assert data.y_lf.shape == (50, 1)
        assert data.x_hf.shape == (5, 1)
        assert data.y_hf.shape == (5, 1)
        assert data.x_test.shape == (100, 1)
        assert data.y_test.shape == (100, 1)

    def test_lf_function(self):
        """Test LF function is identity."""
        data = generate_test1_jump_function()

        # f_L(x) = x
        torch.testing.assert_close(data.y_lf, data.x_lf)

    def test_hf_function(self):
        """Test HF function has jump."""
        data = generate_test1_jump_function(n_test=100)

        # Check discontinuity at x=0
        x = data.x_test
        y = data.y_test

        # For x <= 0: y = x + 1
        mask_neg = x.squeeze() <= 0
        if mask_neg.any():
            expected_neg = x[mask_neg] + 1
            torch.testing.assert_close(y[mask_neg], expected_neg)

        # For x > 0: y = x - 1
        mask_pos = x.squeeze() > 0
        if mask_pos.any():
            expected_pos = x[mask_pos] - 1
            torch.testing.assert_close(y[mask_pos], expected_pos)


class TestTest2NonlinearCorrelation:
    """Tests for Test 2: Nonlinear correlation."""

    def test_output_shape(self):
        """Test output shapes."""
        data = generate_test2_nonlinear_correlation(n_lf=51, n_hf=14, n_test=100)

        assert data.x_lf.shape == (51, 1)
        assert data.x_hf.shape == (14, 1)
        assert data.x_test.shape == (100, 1)

    def test_lf_function(self):
        """Test LF function is sin."""
        data = generate_test2_nonlinear_correlation()

        import numpy as np

        expected = torch.sin(2 * np.pi * data.x_lf)
        torch.testing.assert_close(data.y_lf, expected)

    def test_hf_function(self):
        """Test HF function is sin^2."""
        data = generate_test2_nonlinear_correlation()

        import numpy as np

        expected = torch.sin(2 * np.pi * data.x_test) ** 2
        torch.testing.assert_close(data.y_test, expected)


class TestTest3_2DNonlinear:
    """Tests for Test 3: 2D nonlinear."""

    def test_output_shape(self):
        """Test output shapes."""
        data = generate_test3_2d_nonlinear(n_lf=100, n_hf=20, n_test=25)

        assert data.x_lf.shape == (100, 2)
        assert data.y_lf.shape == (100, 1)
        assert data.x_hf.shape == (20, 2)
        assert data.y_hf.shape == (20, 1)

    def test_domain(self):
        """Test data is in [0, 1]^2."""
        data = generate_test3_2d_nonlinear()

        assert data.x_lf.min() >= 0
        assert data.x_lf.max() <= 1
        assert data.x_hf.min() >= 0
        assert data.x_hf.max() <= 1


class TestTest4HigherDimensional:
    """Tests for Test 4: 4D function."""

    def test_output_shape(self):
        """Test output shapes."""
        data = generate_test4_higher_dimensional(n_lf=100, n_hf=20, n_test=50)

        assert data.x_lf.shape == (100, 4)
        assert data.y_lf.shape == (100, 1)
        assert data.x_hf.shape == (20, 4)
        assert data.y_hf.shape == (20, 1)

    def test_noise_in_lf(self):
        """Test LF data has noise."""
        data = generate_test4_higher_dimensional(noise_std=0.5, seed=42)

        # HF is clean sum of sins
        y_hf_clean = torch.sin(data.x_lf).sum(dim=-1, keepdim=True)

        # LF should be noisy version
        diff = (data.y_lf - y_hf_clean).abs().mean()
        assert diff > 0.1  # Should have some noise


class TestTest5Poisson:
    """Tests for Test 5: Poisson equation."""

    def test_output_shape(self):
        """Test output shapes."""
        data = generate_test5_poisson(n_lf=100, n_hf=20, n_test=50)

        assert data.x_lf.shape == (100, 1)
        assert data.y_lf.shape == (100, 1)
        assert data.x_hf.shape == (20, 1)

    def test_boundary_conditions(self):
        """Test approximate boundary conditions."""
        data = generate_test5_poisson()

        # u(0) and u(1) should be approximately 0
        import numpy as np

        # Analytical: u(x) = sin(pi*x) / pi^2
        # At boundaries: sin(0) = sin(pi) = 0
        x_boundary = torch.tensor([[0.0], [1.0]])
        y_expected = torch.sin(np.pi * x_boundary) / (np.pi ** 2)
        assert y_expected.abs().max() < 0.01


class TestTest7Extrapolation:
    """Tests for Test 7: Extrapolation."""

    def test_output_shape(self):
        """Test output shapes."""
        data = generate_test7_extrapolation(n_lf=100, n_hf=20, n_test=50)

        assert data.x_lf.shape == (100, 1)
        assert data.x_hf.shape == (20, 1)
        assert data.x_test.shape == (50, 1)

    def test_range_difference(self):
        """Test LF covers wider range than HF."""
        data = generate_test7_extrapolation(
            lf_range=(0.0, 2.0),
            hf_range=(0.0, 1.0),
        )

        assert data.x_lf.max() > data.x_hf.max()


class TestGetTestData:
    """Tests for get_test_data function."""

    def test_get_all_tests(self):
        """Test getting all available tests."""
        for test_name in ["test1", "test2", "test3", "test4", "test5", "test7"]:
            data = get_test_data(test_name, n_lf=10, n_hf=5)
            assert isinstance(data, MultifidelityData)

    def test_invalid_test_name(self):
        """Test error for invalid test name."""
        with pytest.raises(ValueError, match="Unknown test"):
            get_test_data("invalid_test")


class TestListTests:
    """Tests for list_tests function."""

    def test_list_tests(self):
        """Test listing available tests."""
        tests = list_tests()

        assert isinstance(tests, dict)
        assert "test1" in tests
        assert "test2" in tests
        assert len(tests) == 7
