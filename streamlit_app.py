import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import os
import datetime

# Cấu hình trang
st.set_page_config(
    page_title="Electric Power Forecasting & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Thiết lập theme và CSS nâng cao
st.markdown("""
    <style>
    /* Màu sắc chính và phụ */
    :root {
        --primary-color: #1E3A8A;
        --primary-light: #3B82F6;
        --primary-dark: #1E40AF;
        --secondary-color: #10B981;
        --accent-color: #F59E0B;
        --text-color: #1F2937;
        --text-light: #6B7280;
        --bg-light: #F9FAFB;
        --bg-card: #FFFFFF;
        --anomaly-high: #EF4444;
        --anomaly-high-bg: #FEE2E2;
        --anomaly-normal: #10B981;
        --anomaly-normal-bg: #ECFDF5;
    }
    
    /* Thiết lập chung */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header và tiêu đề */
    .main-header {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-dark));
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    .main-header p {
        margin-top: 0.5rem;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1v3fvcr {
        background-color: var(--bg-light);
    }
    
    /* Nút */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-dark));
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, var(--primary-dark), var(--primary-color));
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Card metrics */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background-color: var(--bg-card);
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 1.2rem;
        text-align: center;
        flex: 1;
        min-width: 200px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 4px solid var(--primary-color);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-light);
        margin-top: 0;
    }
    
    /* Badges */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .anomaly-high {
        background-color: var(--anomaly-high-bg);
        color: var(--anomaly-high);
    }
    
    .anomaly-normal {
        background-color: var(--anomaly-normal-bg);
        color: var(--anomaly-normal);
    }
    
    /* Slider */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
    }
    
    .stSlider > div > div {
        background-color: var(--primary-light) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 3px solid var(--primary-color) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-card {
            min-width: 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# Hàm cache cho việc tải mô hình
@st.cache_resource
def load_model():
    nf = NeuralForecast.load(path='model_nhits')
    return nf


# Hàm cache cho việc tải và tiền xử lý dữ liệu
@st.cache_data
def load_and_preprocess(train_path, test_path):
    train_df = pd.read_csv(train_path, parse_dates=['NGAYGIO_rounded'])
    test_df = pd.read_csv(test_path, parse_dates=['NGAYGIO_rounded'])

    for df in (train_df, test_df):
        df.rename(columns={
            'SO_CTO': 'unique_id',
            'NGAYGIO_rounded': 'ds',
            'tv delta': 'y'
        }, inplace=True)
        if df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_convert(None)
        df.sort_values(['unique_id', 'ds'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    scalers = {}
    train_scaled, test_scaled = [], []
    for uid, grp in train_df.groupby('unique_id'):
        g_train = grp.copy()
        scaler = MinMaxScaler()
        g_train['y'] = scaler.fit_transform(g_train[['y']])
        scalers[uid] = scaler
        train_scaled.append(g_train)

        g_test = test_df[test_df['unique_id'] == uid].copy()
        if not g_test.empty:
            g_test['y'] = scaler.transform(g_test[['y']])
            test_scaled.append(g_test)

    train_scaled_df = pd.concat(train_scaled).reset_index(drop=True)
    test_scaled_df = pd.concat(test_scaled).reset_index(drop=True)
    return train_scaled_df, test_scaled_df, scalers


# Hàm lấy lịch sử dữ liệu
def get_history(train_df, test_df, unique_id, start_date, input_len):
    train_uid = train_df[train_df['unique_id'] == unique_id].sort_values('ds')
    test_uid = test_df[test_df['unique_id'] == unique_id].sort_values('ds')
    full = pd.concat([train_uid, test_uid]).sort_values('ds').reset_index(drop=True)
    if full['ds'].dt.tz is not None:
        full['ds'] = full['ds'].dt.tz_convert(None)
    idxs = full.index[full['ds'] == pd.to_datetime(start_date)]
    if len(idxs) == 0 or idxs[0] < input_len:
        return None
    idx = idxs[0]
    return full.iloc[idx - input_len:idx].copy()


# Hàm dự đoán cho một ID duy nhất
def predict_single_uid(model, history, uid):
    df_input = history.copy()
    df_input['unique_id'] = uid
    pred = model.predict(df=df_input)
    return pred


# Hàm vẽ biểu đồ dự báo với phát hiện bất thường
def plot_forecast(pred_df, scaler, test_df, start_date, anomaly_threshold=3.0):
    # Kết hợp dữ liệu dự báo và thực tế
    merged = pd.merge(test_df, pred_df, on=['unique_id', 'ds'], how='inner')
    if merged.empty:
        st.warning("Không có dữ liệu khả dụng để hiển thị.")
        return None, None, None
    
    # Chuyển đổi giá trị về thang đo ban đầu
    merged['y_true'] = scaler.inverse_transform(merged[['y']])
    merged['y_pred'] = scaler.inverse_transform(merged[['NHITS']])
    
    # Tính toán sai số và xác định bất thường
    merged['error'] = np.abs(merged['y_true'] - merged['y_pred'])
    threshold = merged['error'].mean() + anomaly_threshold * merged['error'].std()
    merged['is_anomaly'] = merged['error'] > threshold
    
    # Tạo biểu đồ với subplot
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Dự báo & Phát hiện bất thường", "Sai số dự báo")
    )
    
    # Thêm dữ liệu thực tế
    fig.add_trace(
        go.Scatter(
            x=merged['ds'],
            y=merged['y_true'].values,
            mode='lines+markers',
            name='Thực tế',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Thêm dữ liệu dự báo
    fig.add_trace(
        go.Scatter(
            x=merged['ds'],
            y=merged['y_pred'].values,
            mode='lines',
            name='Dự báo',
            line=dict(color='#F59E0B', width=2)
        ),
        row=1, col=1
    )
    
    # Thêm điểm bất thường
    if any(merged['is_anomaly']):
        fig.add_trace(
            go.Scatter(
                x=merged[merged['is_anomaly']]['ds'],
                y=merged[merged['is_anomaly']]['y_true'].values,
                mode='markers',
                name='Bất thường',
                marker=dict(color='#EF4444', size=10, symbol='circle-open', line=dict(width=2))
            ),
            row=1, col=1
        )
    
    # Thêm đường phân cách tại thời điểm bắt đầu dự báo
    start_date_pd = pd.to_datetime(start_date)
    fig.add_vline(
        x=start_date_pd, 
        line_width=2, 
        line_dash="dash", 
        line_color="#6B7280",
        row=1, col=1
    )
    
    # Thêm chú thích cho đường phân cách
    max_y_value = merged['y_true'].max()
    fig.add_annotation(
        x=start_date_pd,
        y=max_y_value * 1.05,
        text="Bắt đầu dự báo",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=-30,
        row=1, col=1
    )
    
    # Thêm biểu đồ sai số
    fig.add_trace(
        go.Bar(
            x=merged['ds'],
            y=merged['error'].values,
            name='Sai số',
            marker=dict(
                color=merged['is_anomaly'].map({True: '#EF4444', False: '#94A3B8'}),
                line=dict(width=0)
            )
        ),
        row=2, col=1
    )
    
    # Thêm đường ngưỡng bất thường
    fig.add_trace(
        go.Scatter(
            x=[merged['ds'].min(), merged['ds'].max()],
            y=[threshold, threshold],
            mode='lines',
            name=f'Ngưỡng bất thường ({anomaly_threshold}σ)',
            line=dict(color='#EF4444', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Cập nhật layout
    fig.update_layout(
        title=dict(
            text="Dự báo tiêu thụ điện & Phát hiện bất thường",
            font=dict(size=20, color='#1E3A8A')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hovermode="x unified",
        height=700,
        template="plotly_white",
        margin=dict(l=10, r=10, t=80, b=10),
    )
    
    # Cập nhật trục x và y
    fig.update_xaxes(
        title_text="Thời gian",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Tiêu thụ điện (kWh)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Thời gian",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Sai số (kWh)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)',
        row=2, col=1
    )
    
    # Tính toán các chỉ số thống kê
    stats = {
        'total_points': len(merged),
        'anomaly_count': merged['is_anomaly'].sum(),
        'anomaly_percent': (merged['is_anomaly'].sum() / len(merged)) * 100,
        'mean_error': merged['error'].mean(),
        'max_error': merged['error'].max(),
        'threshold': threshold
    }
    
    return fig, merged, stats


# Đường dẫn dữ liệu
train_path = 'data/train.csv'
test_path = 'data/test.csv'
input_len = 56
forecast_len = 28

# Header chính
st.markdown("""
<div class="main-header">
    <h1>⚡ Dự báo tiêu thụ điện & Phát hiện bất thường</h1>
    <p>Hệ thống dự báo và phát hiện bất thường trong tiêu thụ điện năng dựa trên mô hình học sâu</p>
</div>
""", unsafe_allow_html=True)

# Tải dữ liệu và mô hình
train_df, test_df, scalers = load_and_preprocess(train_path, test_path)
model = load_model()

# Sidebar
with st.sidebar:
    st.image("image.png", width=300)
    
    st.markdown("### ⚙️ Tham số đầu vào")
    
    # Chọn công tơ
    unique_ids = sorted(test_df['unique_id'].unique())
    uid = st.selectbox("Chọn công tơ (unique_id):", unique_ids)
    
    # Chọn thời điểm bắt đầu dự báo
    start_dates = test_df[test_df['unique_id'] == uid]['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    start_date_str = st.selectbox("Chọn thời điểm bắt đầu dự báo:", start_dates)
    start_date = pd.to_datetime(start_date_str)
    
    # Thêm slider điều chỉnh ngưỡng phát hiện bất thường
    st.markdown("### 🔍 Phát hiện bất thường")
    anomaly_threshold = st.sidebar.slider(
        "Ngưỡng phát hiện bất thường (số lần độ lệch chuẩn)",
        1.0, 5.0, 3.0, 0.1
    )
    
    st.markdown("""
    <div style="margin-top: 10px; font-size: 0.85rem; color: #6B7280;">
        <span style="font-weight: 600;">Lưu ý:</span> Giá trị thấp hơn sẽ phát hiện nhiều bất thường hơn nhưng có thể tăng cảnh báo sai.
    </div>
    """, unsafe_allow_html=True)
    
    # Nút dự báo
    forecast_button = st.button("🔍 Dự báo & Phát hiện bất thường", use_container_width=True)
    
    st.divider()
    st.markdown("### ℹ️ Thông tin")
    st.info(f"Mô hình sử dụng {input_len} điểm dữ liệu lịch sử")
    st.info(f"Độ dài dự báo: {forecast_len} bước")
    st.info("Dự báo 7 ngày tiếp theo từ thời điểm bắt đầu")


# Khu vực chính
if forecast_button:
    with st.spinner("Đang chạy dự báo..."):
        # Lấy dữ liệu lịch sử
        history = get_history(train_df, test_df, uid, start_date, input_len)
        
        if history is None:
            st.error("Không đủ dữ liệu lịch sử để dự báo.")
        else:
            try:
                # Dự báo
                pred = predict_single_uid(model, history, uid)
                
                # Vẽ biểu đồ và tính toán thống kê
                fig, result_df, stats = plot_forecast(
                    pred, 
                    scalers[uid], 
                    test_df[test_df['unique_id'] == uid],
                    start_date,
                    anomaly_threshold
                )
                
                if fig is not None:
                    # Hiển thị các chỉ số thống kê
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{stats['total_points']}</div>
                            <div class="metric-label">Tổng số điểm dữ liệu</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{stats['anomaly_count']}</div>
                            <div class="metric-label">Số điểm bất thường</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{stats['anomaly_percent']:.1f}%</div>
                            <div class="metric-label">Tỷ lệ bất thường</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{stats['mean_error']:.2f}</div>
                            <div class="metric-label">Sai số trung bình (kWh)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Hiển thị biểu đồ
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabs cho thông tin chi tiết
                    tab1, tab2, tab3 = st.tabs(["📊 Dữ liệu chi tiết", "📈 Phân tích bất thường", "ℹ️ Thông tin mô hình"])
                    
                    with tab1:
                        # Hiển thị dữ liệu chi tiết
                        if result_df is not None:
                            # Định dạng dữ liệu
                            display_df = result_df.copy()
                            display_df['ds'] = display_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            display_df['Trạng thái'] = np.where(
                                display_df['is_anomaly'], 
                                '⚠️ Bất thường', 
                                '✅ Bình thường'
                            )
                            
                            # Hiển thị bảng dữ liệu
                            st.dataframe(
                                display_df[['ds', 'y_true', 'y_pred', 'error', 'Trạng thái']].rename(
                                    columns={
                                        'ds': 'Thời gian',
                                        'y_true': 'Thực tế (kWh)',
                                        'y_pred': 'Dự báo (kWh)',
                                        'error': 'Sai số (kWh)'
                                    }
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    with tab2:
                        if result_df is not None and any(result_df['is_anomaly']):
                            # Phân tích bất thường
                            st.markdown("### Phân tích điểm bất thường")
                            
                            # Lọc các điểm bất thường
                            anomalies = result_df[result_df['is_anomaly']].copy()
                            anomalies['ds'] = anomalies['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            anomalies['error_percent'] = (anomalies['error'] / anomalies['y_true']) * 100
                            
                            # Hiển thị bảng bất thường
                            st.dataframe(
                                anomalies[['ds', 'y_true', 'y_pred', 'error', 'error_percent']].rename(
                                    columns={
                                        'ds': 'Thời gian',
                                        'y_true': 'Thực tế (kWh)',
                                        'y_pred': 'Dự báo (kWh)',
                                        'error': 'Sai số tuyệt đối (kWh)',
                                        'error_percent': 'Sai số phần trăm (%)'
                                    }
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Biểu đồ phân bố sai số
                            st.markdown("### Phân bố sai số")
                            error_fig = px.histogram(
                                result_df, 
                                x='error',
                                nbins=20,
                                color='is_anomaly',
                                color_discrete_map={True: '#EF4444', False: '#94A3B8'},
                                labels={'error': 'Sai số (kWh)', 'count': 'Số lượng'},
                                title='Phân bố sai số dự báo'
                            )
                            
                            error_fig.add_vline(
                                x=stats['threshold'], 
                                line_width=2, 
                                line_dash="dash", 
                                line_color="#EF4444",
                                annotation_text=f"Ngưỡng: {stats['threshold']:.2f}"
                            )
                            
                            error_fig.update_layout(
                                showlegend=True,
                                legend_title_text='Bất thường',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(error_fig, use_container_width=True)
                        else:
                            st.info("Không phát hiện bất thường nào trong dữ liệu.")
                    
                    with tab3:
                        # Thông tin mô hình
                        st.markdown("### Thông tin mô hình")
                        st.markdown("""
                        Mô hình dự báo sử dụng kiến trúc **NHITS** (Neural Hierarchical Interpolation for Time Series Forecasting), một mô hình học sâu hiện đại được thiết kế đặc biệt cho dự báo chuỗi thời gian.
                        
                        **Đặc điểm chính:**
                        - Kiến trúc phân cấp với nhiều khối nội suy
                        - Khả năng học các mẫu thời gian ở nhiều tần số khác nhau
                        - Hiệu quả cao với dữ liệu tiêu thụ điện
                        
                        **Phát hiện bất thường:**
                        - Sử dụng phương pháp dựa trên ngưỡng sai số
                        - Ngưỡng được tính bằng: Trung bình sai số + (Độ lệch chuẩn sai số × Hệ số)
                        - Hệ số có thể điều chỉnh để thay đổi độ nhạy của phát hiện
                        """)
                        
                        # Hiển thị thông tin về độ chính xác
                        st.markdown("### Độ chính xác")
                        accuracy_data = {
                            'Metric': ['MAE', 'RMSE', 'MAPE'],
                            'Value': [f"{stats['mean_error']:.4f}", f"{np.sqrt((result_df['error']**2).mean()):.4f}", f"{(result_df['error'] / result_df['y_true']).mean() * 100:.2f}%"]
                        }
                        
                        st.dataframe(pd.DataFrame(accuracy_data), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Lỗi khi chạy dự báo: {str(e)}")
