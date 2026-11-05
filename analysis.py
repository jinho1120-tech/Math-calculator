"""Statistical analysis functions for Scientific Data Analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MODEL_LABELS = {
    "linear": "선형 회귀",
    "polynomial": "다항 회귀",
    "exponential": "지수 회귀",
    "logarithmic": "로그 회귀",
    "power": "거듭제곱 회귀",
}


def _validate_xy(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("X와 Y 데이터의 개수가 다릅니다.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        raise ValueError("유효한 데이터가 최소 2개 필요합니다.")
    if np.unique(x).size < 2:
        raise ValueError("X 데이터에 서로 다른 값이 최소 2개 필요합니다.")
    return x, y


def fit_regression(x, y, model="linear", degree=2):
    """Fit a supported model and return metrics plus a vectorized predictor."""
    x, y = _validate_xy(x, y)
    if model not in MODEL_LABELS:
        raise ValueError(f"지원하지 않는 회귀 모델입니다: {model}")

    p_value = None
    if model == "linear":
        fit = stats.linregress(x, y)
        slope, intercept = float(fit.slope), float(fit.intercept)
        predict = lambda values: slope * np.asarray(values, dtype=float) + intercept
        equation = f"y = {slope:.6g}x {'+' if intercept >= 0 else '-'} {abs(intercept):.6g}"
        p_value = float(fit.pvalue)
    elif model == "polynomial":
        degree = int(degree)
        if degree < 2 or degree > 5:
            raise ValueError("다항식 차수는 2~5 사이여야 합니다.")
        if x.size <= degree:
            raise ValueError(f"{degree}차 다항 회귀에는 데이터가 최소 {degree + 1}개 필요합니다.")
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        predict = lambda values: polynomial(np.asarray(values, dtype=float))
        equation = "y = " + " + ".join(
            f"({coef:.6g})x^{power}" for power, coef in zip(range(degree, -1, -1), coefficients)
        )
    elif model == "exponential":
        if np.any(y <= 0):
            raise ValueError("지수 회귀는 Y 값이 모두 0보다 커야 합니다.")
        fit = stats.linregress(x, np.log(y))
        a, b = float(np.exp(fit.intercept)), float(fit.slope)
        predict = lambda values: a * np.exp(b * np.asarray(values, dtype=float))
        equation = f"y = {a:.6g} · exp({b:.6g}x)"
        p_value = float(fit.pvalue)
    elif model == "logarithmic":
        if np.any(x <= 0):
            raise ValueError("로그 회귀는 X 값이 모두 0보다 커야 합니다.")
        fit = stats.linregress(np.log(x), y)
        a, b = float(fit.intercept), float(fit.slope)
        predict = lambda values: a + b * np.log(np.asarray(values, dtype=float))
        equation = f"y = {a:.6g} + {b:.6g} · ln(x)"
        p_value = float(fit.pvalue)
    else:  # power
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("거듭제곱 회귀는 X와 Y 값이 모두 0보다 커야 합니다.")
        fit = stats.linregress(np.log(x), np.log(y))
        a, b = float(np.exp(fit.intercept)), float(fit.slope)
        predict = lambda values: a * np.power(np.asarray(values, dtype=float), b)
        equation = f"y = {a:.6g} · x^{b:.6g}"
        p_value = float(fit.pvalue)

    predicted = np.asarray(predict(x), dtype=float)
    residuals = y - predicted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - sse / sst) if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    return {
        "predict": predict,
        "equation": equation,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "sse": sse,
        "p_value": p_value,
        "residuals": residuals,
    }


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return common descriptive statistics for numeric columns."""
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["variance"] = numeric.var(ddof=1)
    summary["sem"] = numeric.sem()
    summary["median"] = numeric.median()
    return summary.rename(
        columns={
            "count": "개수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "25%": "제1사분위수",
            "50%": "중앙값",
            "75%": "제3사분위수",
            "max": "최댓값",
            "variance": "분산",
            "sem": "표준오차",
            "median": "중앙값(별도)",
        }
    )
