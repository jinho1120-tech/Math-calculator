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
