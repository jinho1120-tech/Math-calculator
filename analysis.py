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


def _validate_xy(x, y, minimum=2):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("X와 Y 데이터의 개수가 다릅니다.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < minimum:
        raise ValueError(f"유효한 데이터가 최소 {minimum}개 필요합니다.")
    if np.unique(x).size < 2:
        raise ValueError("X 데이터에 서로 다른 값이 최소 2개 필요합니다.")
    return x, y


def _regression_metrics(x, y, predict):
    predicted = np.asarray(predict(x), dtype=float)
    residuals = y - predicted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "r2": float(1 - sse / sst) if sst > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "sse": sse,
        "residuals": residuals,
    }


def fit_regression(x, y, model="linear", degree=2):
    """Fit a supported unweighted model and return metrics plus predictor."""
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
    else:
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("거듭제곱 회귀는 X와 Y 값이 모두 0보다 커야 합니다.")
        fit = stats.linregress(np.log(x), np.log(y))
        a, b = float(np.exp(fit.intercept)), float(fit.slope)
        predict = lambda values: a * np.power(np.asarray(values, dtype=float), b)
        equation = f"y = {a:.6g} · x^{b:.6g}"
        p_value = float(fit.pvalue)

    return {
        "predict": predict,
        "equation": equation,
        "p_value": p_value,
        **_regression_metrics(x, y, predict),
    }


def weighted_linear_regression(x, y, uncertainty):
    """Weighted least squares using w=1/sigma^2, where sigma is 1-SD uncertainty."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.asarray(uncertainty, dtype=float).reshape(-1)
    if not (x.size == y.size == sigma.size):
        raise ValueError("X, Y, 불확도 데이터의 개수가 다릅니다.")
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma)
    x, y, sigma = x[mask], y[mask], sigma[mask]
    if x.size < 3:
        raise ValueError("가중 선형회귀에는 유효한 데이터가 최소 3개 필요합니다.")
    if np.unique(x).size < 2:
        raise ValueError("X 데이터에 서로 다른 값이 최소 2개 필요합니다.")
    if np.any(sigma <= 0):
        raise ValueError("표준불확도(1σ)는 모두 0보다 커야 합니다.")

    weights = 1.0 / np.square(sigma)
    design = np.column_stack((x, np.ones_like(x)))
    root_w = np.sqrt(weights)
    beta, _, rank, _ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
    if rank < 2:
        raise ValueError("가중 회귀 설계행렬의 계수를 계산할 수 없습니다.")
    slope, intercept = map(float, beta)
    predict = lambda values: slope * np.asarray(values, dtype=float) + intercept
    residuals = y - predict(x)
    dof = x.size - 2
    covariance_base = np.linalg.inv(design.T @ (weights[:, None] * design))
    reduced_chi_square = float(np.sum(weights * residuals**2) / dof)
    # sigma is an absolute 1-SD measurement uncertainty; do not rescale covariance.
    slope_se = float(np.sqrt(max(covariance_base[0, 0], 0.0)))
    t_stat = slope / slope_se if slope_se > 0 else float("inf")
    # With supplied absolute standard uncertainties, the slope statistic is
    # standardized by its known-variance standard error (normal reference).
    p_value = float(2 * stats.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else 0.0
    return {
        "predict": predict,
        "equation": f"y = {slope:.6g}x {'+' if intercept >= 0 else '-'} {abs(intercept):.6g}",
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "weights": weights,
        "reduced_chi_square": reduced_chi_square,
        **_regression_metrics(x, y, predict),
    }


def linear_prediction_intervals(x, y, x_new, confidence=0.95):
    """OLS mean-response confidence and individual prediction intervals."""
    x, y = _validate_xy(x, y, minimum=3)
    x_new = np.asarray(x_new, dtype=float)
    if not 0 < confidence < 1:
        raise ValueError("신뢰수준은 0과 1 사이여야 합니다.")
    fit = stats.linregress(x, y)
    y_hat = fit.intercept + fit.slope * x_new
    residuals = y - (fit.intercept + fit.slope * x)
    dof = x.size - 2
    residual_variance = float(np.sum(residuals**2) / dof)
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0:
        raise ValueError("X의 변동이 부족하여 구간을 계산할 수 없습니다.")
    t_critical = float(stats.t.ppf((1 + confidence) / 2, dof))
    mean_se = np.sqrt(residual_variance * (1 / x.size + (x_new - x_mean) ** 2 / sxx))
    prediction_se = np.sqrt(residual_variance * (1 + 1 / x.size + (x_new - x_mean) ** 2 / sxx))
    return {
        "x": x_new,
        "predicted": y_hat,
        "confidence_lower": y_hat - t_critical * mean_se,
        "confidence_upper": y_hat + t_critical * mean_se,
        "prediction_lower": y_hat - t_critical * prediction_se,
        "prediction_upper": y_hat + t_critical * prediction_se,
        "confidence": confidence,
        "degrees_of_freedom": dof,
    }


def correlation_analysis(x, y):
    """Return Pearson and Spearman correlation coefficients and p-values."""
    x, y = _validate_xy(x, y)
    if x.size < 3:
        raise ValueError("상관분석에는 유효한 데이터가 최소 3개 필요합니다.")
    if np.unique(y).size < 2:
        raise ValueError("Y 데이터에 서로 다른 값이 최소 2개 필요합니다.")
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": int(x.size),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return common descriptive statistics for numeric columns."""
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["variance"] = numeric.var(ddof=1)
    summary["sem"] = numeric.sem()
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
        }
    )
