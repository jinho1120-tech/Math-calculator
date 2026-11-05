import numpy as np
import pytest

from residual_diagnostics import residual_diagnostics


def test_linear_diagnostics_return_expected_shapes_and_cutoffs():
    x = np.arange(1, 9, dtype=float)
    y = 2 * x + 1 + np.array([0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.03, -0.03])
    result = residual_diagnostics(x, y, model="linear")
    assert result["residuals"].shape == x.shape
    assert result["leverage"].sum() == pytest.approx(2.0)
    assert result["parameter_count"] == 2
    assert result["degrees_of_freedom"] == 6
    assert result["cook_cutoff"] == pytest.approx(4 / 6)
    assert np.all(np.isfinite(result["studentized_residuals"]))


def test_injected_outlier_is_flagged():
    x = np.arange(1, 11, dtype=float)
    y = 3 * x + 2
    y[-1] += 35
    result = residual_diagnostics(x, y, model="linear")
    assert result["outlier_flags"][-1]
    assert result["studentized_residuals"][-1] > 3


def test_polynomial_diagnostics_use_correct_parameter_count():
    x = np.arange(-4, 5, dtype=float)
    y = x**2 + 2 * x + 3 + np.array([0.1, -0.1, 0.05, 0, -0.05, 0.08, -0.08, 0.03, -0.03])
    result = residual_diagnostics(x, y, model="polynomial", degree=2)
    assert result["parameter_count"] == 3
    assert result["leverage"].sum() == pytest.approx(3.0)
    assert result["shapiro_p_value"] >= 0


def test_exact_fit_has_zero_standardized_residuals():
    x = np.arange(1, 6, dtype=float)
    result = residual_diagnostics(x, 4 * x - 2, model="linear")
    assert np.allclose(result["standardized_residuals"], 0)
    assert np.allclose(result["cooks_distance"], 0)


def test_diagnostics_reject_unsupported_models_and_small_samples():
    with pytest.raises(ValueError):
        residual_diagnostics([1, 2, 3], [2, 4, 6], model="exponential")
    with pytest.raises(ValueError):
        residual_diagnostics([1, 2], [2, 4], model="linear")
