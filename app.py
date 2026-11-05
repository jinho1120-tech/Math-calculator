import io
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    MODEL_LABELS,
    correlation_analysis,
    descriptive_statistics,
    fit_regression,
    linear_prediction_intervals,
    weighted_linear_regression,
)

st.set_page_config(page_title="Scientific Data Analyzer", page_icon="🔬", layout="wide")
st.title("🔬 Scientific Data Analyzer")
st.caption("논문 및 실험 데이터 분석을 위한 종합 과학 계산 플랫폼")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        {"X": [1., 2., 3., 4., 5.], "Y": [2.1, 3.9, 6.2, 7.8, 10.1]}
    )
if "data_version" not in st.session_state:
    st.session_state.data_version = 0
if "loaded_file_signature" not in st.session_state:
    st.session_state.loaded_file_signature = None

with st.sidebar:
    st.header("데이터 입력")
    uploaded = st.file_uploader("CSV 또는 Excel 파일", type=["csv", "xlsx"])
    if uploaded is not None:
        signature = (uploaded.name, hashlib.sha256(uploaded.getvalue()).hexdigest())
        if signature != st.session_state.loaded_file_signature:
            try:
                loaded = (pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv")
                          else pd.read_excel(uploaded))
                st.session_state.data = loaded
                st.session_state.data_version += 1
                st.session_state.loaded_file_signature = signature
                st.success(f"{len(loaded):,}개 행을 불러왔습니다.")
            except Exception as exc:
                st.error(f"파일을 읽지 못했습니다: {exc}")
    if st.button("예제 데이터로 초기화", use_container_width=True):
        st.session_state.data = pd.DataFrame(
            {"X": [1., 2., 3., 4., 5.], "Y": [2.1, 3.9, 6.2, 7.8, 10.1]}
        )
        st.session_state.data_version += 1
        st.session_state.loaded_file_signature = None
        st.rerun()

st.subheader("1. 데이터")
st.caption("표에 직접 입력하거나 Excel/CSV 데이터를 붙여넣을 수 있습니다.")
edited = st.data_editor(
    st.session_state.data, num_rows="dynamic", use_container_width=True,
    key=f"data_editor_{st.session_state.data_version}",
)
st.session_state.data = edited
numeric_columns = edited.select_dtypes(include="number").columns.tolist()
if len(numeric_columns) < 2:
    st.info("분석하려면 숫자형 열이 최소 2개 필요합니다.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    x_col = st.selectbox("독립변수 (X)", numeric_columns, index=0)
with c2:
    y_col = st.selectbox("종속변수 (Y)", numeric_columns, index=1)
with c3:
    uncertainty_options = ["사용 안 함"] + [c for c in numeric_columns if c not in (x_col, y_col)]
    uncertainty_col = st.selectbox("Y 표준불확도 (1σ) 열", uncertainty_options)
if x_col == y_col:
    st.warning("X와 Y에는 서로 다른 열을 선택하세요.")
    st.stop()

selected = [x_col, y_col] + ([uncertainty_col] if uncertainty_col != "사용 안 함" else [])
clean = edited[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
x = clean[x_col].to_numpy(dtype=float)
y = clean[y_col].to_numpy(dtype=float)
sigma = clean[uncertainty_col].to_numpy(dtype=float) if uncertainty_col != "사용 안 함" else None
st.caption(f"유효 데이터: {len(clean):,}개 / 제외된 행: {len(edited) - len(clean):,}개")
if len(clean) < 2:
    st.warning("회귀분석에는 유효한 데이터가 최소 2개 필요합니다.")
    st.stop()

st.subheader("2. 회귀분석")
mcol, dcol = st.columns([2, 1])
with mcol:
    model = st.selectbox("회귀 모델", list(MODEL_LABELS), format_func=lambda key: MODEL_LABELS[key])
with dcol:
    degree = st.selectbox("다항식 차수", [2, 3, 4, 5], disabled=model != "polynomial")
if sigma is not None and np.any(sigma <= 0):
    st.error("표준불확도 열에는 0보다 큰 값만 입력하세요.")
    st.stop()
if sigma is not None and model != "linear":
    st.warning("표준불확도 열을 선택했으므로 오차막대는 표시하지만, 가중회귀는 선형 모델에서만 적용합니다.")

try:
    if sigma is not None and model == "linear":
        result = weighted_linear_regression(x, y, sigma)
        weighted = True
    else:
        result = fit_regression(x, y, model=model, degree=degree)
        weighted = False
except (ValueError, np.linalg.LinAlgError) as exc:
    st.error(str(exc))
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{result['r2']:.5f}")
m2.metric("RMSE", f"{result['rmse']:.5g}")
m3.metric("MAE", f"{result['mae']:.5g}")
m4.metric("데이터 수", f"{len(x):,}")
st.latex(result["equation"])
if result.get("p_value") is not None:
    label = "가중회귀 기울기 p-value" if weighted else "모델 변환 공간 기울기 p-value"
    st.caption(f"{label}: {result['p_value']:.5g}")
if weighted:
    st.caption("가중치 정의: w = 1/σ² (σ는 각 측정값의 표준불확도, 1σ). 축소 카이제곱은 잔차로 추정한 분산 보정값입니다.")
    st.metric("축소 카이제곱", f"{result['reduced_chi_square']:.5g}")
elif model in ("exponential", "logarithmic", "power"):
    st.caption("비선형 변환 모델의 p-value는 변환된 회귀 공간의 기울기 검정값이며, 원래 Y 척도의 비선형 회귀 검정값이 아닙니다.")

x_line = np.linspace(float(np.min(x)), float(np.max(x)), 300)
y_line = result["predict"](x_line)
residuals = y - result["predict"](x)
fig = go.Figure()
error_y = ({"type": "data", "array": sigma, "visible": True, "thickness": 1.2}
           if sigma is not None else None)
fig.add_trace(go.Scatter(
    x=x, y=y, mode="markers", name="측정 데이터",
    marker={"size": 9, "opacity": 0.8},
    error_y=error_y,
))
fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="회귀 모델", line={"width": 3}))

show_intervals = False
if model == "linear" and not weighted and len(x) >= 3:
    show_intervals = st.checkbox("95% 신뢰구간 및 예측구간 표시", value=True)
    if show_intervals:
        intervals = linear_prediction_intervals(x, y, x_line, confidence=0.95)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([intervals["confidence_upper"], intervals["confidence_lower"][::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.16)",
            line={"color": "rgba(31,119,180,0)"}, name="평균반응 95% 신뢰구간",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([intervals["prediction_upper"], intervals["prediction_lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.10)",
            line={"color": "rgba(255,127,14,0)"}, name="개별값 95% 예측구간",
        ))
fig.update_layout(
    title=f"{y_col} vs {x_col} — {MODEL_LABELS[model]}",
    xaxis_title=x_col, yaxis_title=y_col, hovermode="closest",
    legend={"orientation": "h", "y": 1.15}, margin={"t": 85, "r": 25, "b": 45, "l": 55},
)
st.plotly_chart(fig, use_container_width=True)
if show_intervals:
    st.caption("구간은 통상적인 최소제곱 선형회귀의 정규·등분산 오차 가정에 따른 점별 95% 구간입니다. 데이터 범위 밖 외삽에는 사용하지 마세요.")

with st.expander("회귀 결과 상세 정보"):
    detail = pd.DataFrame({
        "항목": ["모델", "R²", "RMSE", "MAE", "잔차 제곱합 (SSE)"],
        "값": [MODEL_LABELS[model], result["r2"], result["rmse"], result["mae"], result["sse"]],
    })
    st.dataframe(detail, hide_index=True, use_container_width=True)

st.subheader("3. 상관분석")
try:
    corr = correlation_analysis(x, y)
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Pearson r", f"{corr['pearson_r']:.5f}")
    cc2.metric("Pearson p-value", f"{corr['pearson_p']:.4g}")
    cc3.metric("Spearman ρ", f"{corr['spearman_rho']:.5f}")
    cc4.metric("Spearman p-value", f"{corr['spearman_p']:.4g}")
    st.caption("Pearson은 선형 상관, Spearman은 순위 기반 단조 관계를 측정합니다. 상관관계가 인과관계를 뜻하지는 않습니다.")
except ValueError as exc:
    st.info(str(exc))

st.subheader("4. 기술통계")
descriptive = descriptive_statistics(edited)
st.dataframe(descriptive, use_container_width=True)

with st.expander("5. 잔차 분석"):
    residual_fig = go.Figure()
    residual_fig.add_trace(go.Scatter(x=result["predict"](x), y=residuals, mode="markers", name="잔차"))
    residual_fig.add_hline(y=0, line_dash="dash")
    residual_fig.update_layout(title="예측값 대비 잔차", xaxis_title="예측값", yaxis_title="잔차 (관측값 - 예측값)")
    st.plotly_chart(residual_fig, use_container_width=True)
    residual_table = clean.copy()
    residual_table["예측값"] = result["predict"](x)
    residual_table["잔차"] = residuals
    st.dataframe(residual_table, use_container_width=True)

st.subheader("6. 결과 내보내기")
export_df = clean.copy()
export_df["예측값"] = result["predict"](x)
export_df["잔차"] = residuals
if weighted:
    export_df["가중치 (1/σ²)"] = result["weights"]
csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("분석 데이터 CSV 다운로드", data=csv_bytes, file_name="analysis_results.csv", mime="text/csv")
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    export_df.to_excel(writer, sheet_name="Regression", index=False)
    descriptive.to_excel(writer, sheet_name="Descriptive statistics")
    detail.to_excel(writer, sheet_name="Model summary", index=False)
st.download_button(
    "분석 결과 Excel 다운로드", data=excel_buffer.getvalue(),
    file_name="analysis_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
