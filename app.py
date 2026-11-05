import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="종합 추세선 분석기", layout="wide")
st.title("📈 종합 추세선 분석기 (파일 업로드 + 직접 입력)")

# 1. 초기 빈 데이터프레임 생성 (세션 상태 활용)
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({"X": [None] * 5, "Y": [None] * 5}, dtype=float)

# 2. 파일 업로드 시 실행될 콜백 함수
def on_file_upload():
    uploaded_file = st.session_state.file_uploader_key
    if uploaded_file is not None:
        try:
            # 확장자에 따라 다르게 읽기
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            
            # 읽어온 데이터를 세션 상태에 저장하여 에디터에 반영
            st.session_state.df = df
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

# 3. 사이드바: 파일 업로드 기능
st.sidebar.header("📁 데이터 업로드")
st.sidebar.file_uploader(
    "CSV 또는 엑셀(XLSX) 파일을 업로드하세요.", 
    type=["csv", "xlsx"], 
    key="file_uploader_key", 
    on_change=on_file_upload # 파일이 올라가면 on_file_upload 함수 자동 실행
)

st.write("사이드바에서 파일을 업로드하거나, 아래 표에 직접 데이터를 입력(또는 복사/붙여넣기) 하세요. 표의 값을 수정하면 그래프가 실시간으로 반영됩니다.")

# 4. 상호작용형 데이터 에디터 렌더링
edited_df = st.data_editor(
    st.session_state.df, 
    num_rows="dynamic", 
    use_container_width=True,
    hide_index=True
)

# 5. 시각화를 위한 변수 선택 및 데이터 전처리
numeric_columns = edited_df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) < 2:
    st.warning("💡 추세선을 그리려면 표 안에 숫자형 데이터 열이 2개 이상 있어야 합니다.")
else:
    # 컬럼이 여러 개일 경우 X축, Y축 선택
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X축 변수 선택", numeric_columns, index=0)
    with col2:
        default_y_index = 1 if len(numeric_columns) > 1 else 0
        y_col = st.selectbox("Y축 변수 선택", numeric_columns, index=default_y_index)

    # 선택한 두 열에서 빈 칸이 있는 행 제거
    plot_df = edited_df.dropna(subset=[x_col, y_col])

    if len(plot_df) < 2:
        st.info("💡 선택한 변수 열에 유효한 숫자 데이터가 2개 이상 입력되어야 합니다.")
    else:
        x = plot_df[x_col]
        y = plot_df[y_col]

        # 6. 추세선(선형 회귀) 계산
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        y_trend = np.polyval(coefficients, x)

        # 7. 결과 출력 및 시각화
        st.write("---")
        st.subheader("💡 도출된 추세선 방정식")
        
        sign = "+" if intercept >= 0 else "-"
        st.latex(f"y = {slope:.4f}x {sign} {abs(intercept):.4f}")

        # Plotly 그래프 생성
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y, 
            mode='markers', 
            name='실제 데이터',
            marker=dict(color='blue', opacity=0.6, size=10)
        ))

        fig.add_trace(go.Scatter(
            x=x, y=y_trend, 
            mode='lines', 
            name=f'추세선 (y={slope:.2f}x{sign}{abs(intercept):.2f})',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title=f"{x_col} vs {y_col} 산점도 및 선형 추세선",
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode="closest"
        )

        st.plotly_chart(fig, use_container_width=True)
