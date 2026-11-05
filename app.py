import io
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    MODEL_LABELS,
    descriptive_statistics,
    fit_regression,
)

st.set_page_config(
    page_title="Scientific Data Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Scientific Data Analyzer")
st.caption("논문 및 실험 데이터 분석을 위한 종합 과학 계산 플랫폼")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        {"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [2.1, 3.9, 6.2, 7.8, 10.1]}
    )
if "data_version" not in st.session_state:
    st.session_state.data_version = 0
if "loaded_file_signature" not in st.session_state:
    st.session_state.loaded_file_signature = None

with st.sidebar:
    st.header("데이터 입력")
    uploaded = st.file_uploader(
        "CSV 또는 Excel 파일", type=["csv", "xlsx"]
    )
    if uploaded is not None:
        signature = (uploaded.name, hashlib.sha256(uploaded.getvalue()).hexdigest())
        if signature != st.session_state.loaded_file_signature:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    loaded = pd.read_csv(uploaded)
                else:
                    loaded = pd.read_excel(uploaded)
                st.session_state.data = loaded
                st.session_state.data_version += 1
                st.session_state.loaded_file_signature = signature
                st.success(f"{len(loaded):,}개 행을 불러왔습니다.")
            except Exception as exc:
                st.error(f"파일을 읽지 못했습니다: {exc}")

    if st.button("예제 데이터로 초기화", use_container_width=True):
        st.session_state.data = pd.DataFrame(
            {"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [2.1, 3.9, 6.2, 7.8, 10.1]}
        )
        st.session_state.data_version += 1
        st.session_state.loaded_file_signature = None
        st.rerun()

st.subheader("1. 데이터")
st.caption("표에 직접 입력하거나 Excel/CSV 데이터를 붙여넣을 수 있습니다.")
edited = st.data_editor(
    st.session_state.data,
    num_rows="dynamic",
    use_container_width=True,
    key=f"data_editor_{st.session_state.data_version}",
)
st.session_state.data = edited

numeric_columns = edited.select_dtypes(include="number").columns.tolist()
if len(numeric_columns) < 2:
    st.info("분석하려면 숫자형 열이 최소 2개 필요합니다.")
    st.stop()

left, right = st.columns(2)
with left:
    x_col = st.selectbox("독립변수 (X)", numeric_columns, index=0)
with right:
    y_col = st.selectbox(
        "종속변수 (Y)",
        numeric_columns,
        index=1 if len(numeric_columns) > 1 else 0,
    )

if x_col == y_col:
    st.warning("X와 Y에는 서로 다른 열을 선택하세요.")
    st.stop()

clean = edited[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
x = clean[x_col].to_numpy(dtype=float)
y = clean[y_col].to_numpy(dtype=float)

st.caption(f"유효 데이터: {len(clean):,}개 / 제외된 행: {len(edited) - len(clean):,}개")
if len(clean) < 2:
    st.warning("회귀분석에는 유효한 데이터가 최소 2개 필요합니다.")
    st.stop()

st.subheader("2. 회귀분석")
model_col, degree_col = st.columns([2, 1])
with model_col:
    model = st.selectbox(
        "회귀 모델",
        list(MODEL_LABELS),
        format_func=lambda key: MODEL_LABELS[key],
    )
with degree_col:
    degree = st.selectbox("다항식 차수", [2, 3, 4, 5], index=0, disabled=model != "polynomial")

try:
    result = fit_regression(x, y, model=model, degree=degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"모델 계산 중 오류가 발생했습니다: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("R²", f"{result['r2']:.5f}")
m2.metric("RMSE", f"{result['rmse']:.5g}")
m3.metric("MAE", f"{result['mae']:.5g}")
m4.metric("데이터 수", f"{len(x):,}")

st.latex(result["equation"])
if result["p_value"] is not None:
    st.caption(
        f"모델 변환 공간에서 계산한 기울기 p-value: {result['p_value']:.5g}. "
        "비선형 변환 모델의 p-value는 원래 Y 척도의 비선형 회귀 검정값이 아닙니다."
    )
elif model == "polynomial":
    st.caption("다항 회귀의 개별 계수 유의성 검정은 아직 제공하지 않습니다.")

x_line = np.linspace(float(np.min(x)), float(np.max(x)), 300)
y_line = result["predict"](x_line)
order = np.argsort(x)
residuals = y - result["predict"](x)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x, y=y, mode="markers", name="측정 데이터",
        marker={"size": 9, "opacity": 0.8},
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line, y=y_line, mode="lines", name="회귀 모델",
        line={"width": 3},
    )
)
fig.update_layout(
    title=f"{y_col} vs {x_col} — {MODEL_LABELS[model]}",
    xaxis_title=x_col,
    yaxis_title=y_col,
    hovermode="closest",
    legend={"orientation": "h", "y": 1.12},
    margin={"t": 75, "r": 25, "b": 45, "l": 55},
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("회귀 결과 상세 정보"):
    detail = pd.DataFrame(
        {
            "항목": ["모델", "R²", "RMSE", "MAE", "잔차 제곱합 (SSE)"],
            "값": [
                MODEL_LABELS[model],
                result["r2"],
                result["rmse"],
                result["mae"],
                result["sse"],
            ],
        }
    )
    st.dataframe(detail, hide_index=True, use_container_width=True)

st.subheader("3. 기술통계")
stats = descriptive_statistics(clean)
st.dataframe(stats, use_container_width=True)

with st.expander("4. 잔차 분석"):
    residual_fig = go.Figure()
    residual_fig.add_trace(
        go.Scatter(
            x=result["predict"](x), y=residuals,
            mode="markers", name="잔차",
        )
    )
    residual_fig.add_hline(y=0, line_dash="dash")
    residual_fig.update_layout(
        title="예측값 대비 잔차",
        xaxis_title="예측값",
        yaxis_title="잔차 (관측값 - 예측값)",
    )
    st.plotly_chart(residual_fig, use_container_width=True)

    residual_table = clean.copy()
    residual_table["예측값"] = result["predict"](x)
    residual_table["잔차"] = residuals
    st.dataframe(residual_table, use_container_width=True)

st.subheader("5. 결과 내보내기")
export_df = clean.copy()
export_df["예측값"] = result["predict"](x)
export_df["잔차"] = residuals
csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "분석 데이터 CSV 다운로드",
    data=csv_bytes,
    file_name="analysis_results.csv",
    mime="text/csv",
)

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    export_df.to_excel(writer, sheet_name="Regression", index=False)
    stats.to_excel(writer, sheet_name="Descriptive statistics")
    detail.to_excel(writer, sheet_name="Model summary", index=False)
st.download_button(
    "분석 결과 Excel 다운로드",
    data=excel_buffer.getvalue(),
    file_name="analysis_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
