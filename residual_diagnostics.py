"""Residual diagnostics for ordinary least-squares linear and polynomial models."""

from __future__ import annotations

import numpy as np
from scipy import stats


def residual_diagnostics(x, y, model="linear", degree=2):
    """Calculate OLS residual diagnostics for linear/polynomial models.

    Outlier and influence cutoffs are screening heuristics, not automatic
    grounds for excluding observations.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("X와 Y 데이터의 개수가 다릅니다.")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if model == "linear":
        degree = 1
    elif model == "polynomial":
        degree = int(degree)
        if degree < 2 or degree > 5:
            raise ValueError("다항식 차수는 2~5 사이여야 합니다.")
    else:
        raise ValueError("잔차 영향도 분석은 선형 및 다항 회귀에서만 지원합니다.")

    parameter_count = degree + 1
    if x.size <= parameter_count:
        raise ValueError(
            f"{degree}차 회귀 진단에는 유효한 데이터가 최소 {parameter_count + 1}개 필요합니다."
        )
    if np.unique(x).size < parameter_count:
        raise ValueError("회귀 설계행렬의 독립적인 X 값이 충분하지 않습니다.")

    design = np.vander(x, N=parameter_count, increasing=True)
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < parameter_count:
        raise ValueError("회귀 설계행렬의 계수를 계산할 수 없습니다.")

    fitted = design @ coefficients
    residuals = y - fitted
    q, _ = np.linalg.qr(design, mode="reduced")
    leverage = np.sum(q**2, axis=1)
    leverage = np.clip(leverage, 0.0, 1.0)
    dof = int(x.size - parameter_count)
    sse = float(np.sum(residuals**2))
    mse = sse / dof
    residual_scale = float(np.sqrt(mse))
    # Treat floating-point roundoff from an exact fit as zero residual variance.
    numerical_tolerance = np.finfo(float).eps * max(1.0, float(np.linalg.norm(y))) * 10
    if residual_scale <= numerical_tolerance:
        residuals = np.zeros_like(residuals)
        sse = 0.0
        mse = 0.0
        residual_scale = 0.0

    if residual_scale > 0:
        internal = residuals / (residual_scale * np.sqrt(np.maximum(1 - leverage, 0)))
    else:
        internal = np.zeros_like(residuals)

    # Deleted (externally studentized) residuals, when a deleted variance
    # estimate is available. These are more suitable for outlier screening.
    external = np.full_like(residuals, np.nan)
    if dof > 1:
        deleted_sse = sse - np.divide(
            residuals**2, np.maximum(1 - leverage, np.finfo(float).eps)
        )
        deleted_variance = np.maximum(deleted_sse / (dof - 1), 0.0)
        denominator = np.sqrt(deleted_variance * np.maximum(1 - leverage, 0))
        valid = denominator > 0
        external[valid] = residuals[valid] / denominator[valid]
        zero_scale = (denominator == 0) & (residuals != 0)
        external[zero_scale] = np.sign(residuals[zero_scale]) * np.inf
        exact_fit = (residuals == 0) & (denominator == 0)
        external[exact_fit] = 0.0

    cook_distance = np.divide(
        internal**2 * leverage,
        parameter_count * np.maximum(1 - leverage, np.finfo(float).eps),
    )
    outlier_cutoff = 3.0
    leverage_cutoff = 2.0 * parameter_count / x.size
    cook_cutoff = 4.0 / dof

    shapiro_statistic = float("nan")
    shapiro_p_value = float("nan")
    if 3 <= x.size <= 5000 and np.unique(residuals).size > 1:
        shapiro = stats.shapiro(residuals)
        shapiro_statistic = float(shapiro.statistic)
        shapiro_p_value = float(shapiro.pvalue)

    return {
        "x": x,
        "observed": y,
        "fitted": fitted,
        "residuals": residuals,
        "standardized_residuals": internal,
        "studentized_residuals": external,
        "leverage": leverage,
        "cooks_distance": cook_distance,
        "outlier_flags": ~np.isnan(external) & (np.abs(external) > outlier_cutoff),
        "high_leverage_flags": leverage > leverage_cutoff,
        "influential_flags": cook_distance > cook_cutoff,
        "outlier_cutoff": outlier_cutoff,
        "leverage_cutoff": leverage_cutoff,
        "cook_cutoff": cook_cutoff,
        "parameter_count": parameter_count,
        "degrees_of_freedom": dof,
        "shapiro_w": shapiro_statistic,
        "shapiro_p_value": shapiro_p_value,
    }
