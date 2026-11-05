import numpy as np
import pandas as pd
import pytest

from analysis import descriptive_statistics, fit_regression


def test_linear_regression_recovers_line():
    result = fit_regression([1, 2, 3, 4], [3, 5, 7, 9], "linear")
    assert result["r2"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["predict"]([5])[0] == pytest.approx(11.0)


def test_polynomial_regression():
    x = np.arange(1, 6, dtype=float)
    y = x**2 + 2*x + 1
    result = fit_regression(x, y, "polynomial", degree=2)
    assert result["r2"] == pytest.approx(1.0)
    assert result["predict"]([6])[0] == pytest.approx(49.0)


def test_exponential_regression():
    x = np.arange(5, dtype=float)
    y = 2 * np.exp(0.3 * x)
    result = fit_regression(x, y, "exponential")
    assert result["r2"] == pytest.approx(1.0)
    assert result["predict"]([5])[0] == pytest.approx(2 * np.exp(1.5))


def test_invalid_log_input():
    with pytest.raises(ValueError):
        fit_regression([-1, 1, 2], [1, 2, 3], "logarithmic")


def test_descriptive_statistics():
    result = descriptive_statistics(pd.DataFrame({"X": [1, 2, 3]}))
    assert result.loc["X", "평균"] == pytest.approx(2.0)
    assert result.loc["X", "표준편차"] == pytest.approx(1.0)


def test_weighted_linear_regression_recovers_line():
    from analysis import weighted_linear_regression
    result = weighted_linear_regression([1, 2, 3, 4], [3, 5, 7, 9], [0.1, 0.2, 0.1, 0.3])
    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(1.0)
    assert result["predict"]([5])[0] == pytest.approx(11.0)
    assert result["weights"][0] == pytest.approx(100.0)


def test_weighted_regression_rejects_nonpositive_uncertainty():
    from analysis import weighted_linear_regression
    with pytest.raises(ValueError):
        weighted_linear_regression([1, 2, 3], [2, 4, 6], [0.1, 0, 0.2])


def test_linear_confidence_and_prediction_intervals():
    from analysis import linear_prediction_intervals
    x = np.arange(1, 8, dtype=float)
    y = 2 * x + np.array([0.2, -0.1, 0.1, -0.2, 0.15, -0.05, 0.1])
    result = linear_prediction_intervals(x, y, [4.0])
    assert result["confidence_lower"][0] < result["predicted"][0]
    assert result["predicted"][0] < result["confidence_upper"][0]
    assert result["prediction_lower"][0] < result["confidence_lower"][0]
    assert result["prediction_upper"][0] > result["confidence_upper"][0]


def test_pearson_and_spearman_correlation():
    from analysis import correlation_analysis
    result = correlation_analysis([1, 2, 3, 4], [1, 4, 9, 16])
    assert result["pearson_r"] > 0.9
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["n"] == 4
