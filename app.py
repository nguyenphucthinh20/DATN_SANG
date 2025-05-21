import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta
from neuralforecast import NeuralForecast
import plotly.graph_objects as go
import plotly.express as px

# Thiết lập trang
st.set_page_config(
    page_title="Dự báo sản lượng điện & phát hiện bất thường",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tùy chỉnh CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .css-1v0mbdj {
        margin-top: -60px;
    }
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
        color: #1E3A8A;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .css-1v3fvcr {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0;
    }
    .anomaly-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .anomaly-normal {
        background-color: #ECFDF5;
        color: #047857;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# --------- FUNCTIONS ----------

@st.cache_resource
def load_model_and_scalers(model_path, scalers_path):
    """Tải mô hình và scalers (có cache)"""
    try:
        # Load mô hình
        model = NeuralForecast.load(model_path)
        
        # Load scalers
        scalers = joblib.load(scalers_path)
        
        return model, scalers
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình và scalers: {e}")
        st.stop()

@st.cache_data
def load_data(csv_path):
    """Tải dữ liệu ban đầu (có cache)"""
    try:
        df = pd.read_csv(csv_path, parse_dates=['NGAYGIO_rounded'])
        df = df.rename(columns={
            'SO_CTO': 'unique_id',
            'NGAYGIO_rounded': 'ds',
            'tv delta': 'y'
        })
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.stop()

def get_unique_ids(df):
    """Lấy danh sách unique_id từ DataFrame"""
    return sorted(df['unique_id'].unique())

def prepare_data_for_prediction(df, unique_id):
    """Chuẩn bị dữ liệu cho một ID cụ thể"""
    data = df[df['unique_id'] == unique_id].copy()
    
    if len(data) < 10:  # Kiểm tra xem có đủ dữ liệu không
        st.warning(f"Không đủ dữ liệu cho công tơ {unique_id}")
        return None
        
    return data

def scale_data(data, scalers, unique_id):
    """Chuẩn hóa dữ liệu với scalers đã lưu"""
    try:
        if unique_id in scalers:
            scaler = scalers[unique_id]
            data_copy = data.copy()
            data_copy['y'] = scaler.transform(data_copy[['y']])
            return data_copy, scaler
        else:
            st.warning(f"Không tìm thấy scaler cho ID: {unique_id}")
            return None, None
    except Exception as e:
        st.error(f"Lỗi khi chuẩn hóa dữ liệu: {e}")
        return None, None

def make_forecast(model, data, forecast_len):
    """Thực hiện dự báo"""
    try:
        # Đảm bảo rằng dữ liệu đầu vào có đúng định dạng
        train_data = data[['unique_id', 'ds', 'y']].copy()
        
        # Dự báo
        forecast_df = model.predict(df=train_data)
        return forecast_df
    except Exception as e:
        st.error(f"Lỗi khi dự báo: {e}")
        return None

def generate_future_dates(last_date, num_days):
    """Tạo các ngày giờ tương lai để dự báo với các mốc thời gian 00, 06, 12, 18"""
    future_dates = []
    
    # Đảm bảo last_date được chuẩn hóa về múi giờ UTC
    if last_date.tzinfo is not None:
        last_date = last_date.replace(tzinfo=None)
    
    # Tìm mốc thời gian tiếp theo (00, 06, 12, 18)
    current_hour = last_date.hour
    next_time_slot = (current_hour // 6 + 1) * 6
    if next_time_slot == 24:
        # Chuyển sang ngày hôm sau ở múi giờ 00
        start_date = (last_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Chuyển sang mốc giờ tiếp theo trong ngày
        start_date = last_date.replace(hour=next_time_slot, minute=0, second=0, microsecond=0)
    
    # Tạo các mốc thời gian cho số ngày cần dự báo
    for day in range(num_days):
        for hour in [0, 6, 12, 18]:
            # Tính toán ngày và giờ cho mỗi điểm dự báo
            forecast_date = start_date + timedelta(days=day)
            forecast_date = forecast_date.replace(hour=hour)
            future_dates.append(forecast_date)
    
    # Loại bỏ các mốc thời gian đã qua so với start_date
    future_dates = [date for date in future_dates if date >= start_date]
    
    # Chỉ lấy đúng số điểm cần thiết (4 điểm/ngày * số ngày)
    return future_dates[:num_days * 4]

def prepare_future_data(data, unique_id, num_days, start_date=None, start_hour=0):
    """Chuẩn bị dữ liệu cho dự báo tương lai từ thời điểm xác định"""
    
    # Nếu có ngày bắt đầu được chỉ định, sử dụng nó
    if start_date is not None:
        # Tạo datetime từ ngày và giờ được chọn
        start_datetime = datetime.combine(start_date, datetime.min.time())
        start_datetime = start_datetime.replace(hour=start_hour)
    else:
        # Nếu không, sử dụng ngày cuối cùng từ dữ liệu hiện có
        last_date = data['ds'].max()
        
        # Tìm mốc thời gian tiếp theo (00, 06, 12, 18)
        current_hour = last_date.hour
        next_time_slot = (current_hour // 6 + 1) * 6
        if next_time_slot == 24:
            # Chuyển sang ngày hôm sau ở múi giờ 00
            start_datetime = (last_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Chuyển sang mốc giờ tiếp theo trong ngày
            start_datetime = last_date.replace(hour=next_time_slot, minute=0, second=0, microsecond=0)
    
    # Tạo các mốc thời gian cho số ngày cần dự báo
    future_dates = []
    for day in range(num_days):
        for hour in [0, 6, 12, 18]:
            # Tính toán ngày và giờ cho mỗi điểm dự báo
            forecast_date = start_datetime + timedelta(days=day)
            forecast_date = forecast_date.replace(hour=hour)
            future_dates.append(forecast_date)
    
    # Loại bỏ các mốc thời gian trước thời điểm bắt đầu
    future_dates = [date for date in future_dates if date >= start_datetime]
    
    # Chỉ lấy đúng số điểm cần thiết (4 điểm/ngày * số ngày)
    future_dates = future_dates[:num_days * 4]
    
    # Tạo DataFrame cho dữ liệu tương lai
    future_df = pd.DataFrame({
        'unique_id': [unique_id] * len(future_dates),
        'ds': future_dates,
        'y': [np.nan] * len(future_dates)  # NaN vì chưa có giá trị thực tế
    })
    
    return future_df

def detect_anomalies(actual, predicted, scaler, threshold_std=3):
    """Phát hiện bất thường dựa trên sai số giữa giá trị thực tế và dự báo"""
    try:
        # Inverse transform để có giá trị thực
        y_true = scaler.inverse_transform(actual[['y']])
        y_pred = scaler.inverse_transform(predicted[['NHITS']])
        
        # Tính toán sai số
        error = np.abs(y_true - y_pred)
        
        # Tính ngưỡng phát hiện bất thường
        err_mean = np.mean(error)
        err_std = np.std(error)
        threshold = err_mean + threshold_std * err_std
        
        # Đánh dấu bất thường
        anomalies = error > threshold
        
        # Tạo DataFrame kết quả
        result = pd.DataFrame({
            'ds': actual['ds'].values,
            'y_true': y_true.flatten(),
            'y_pred': y_pred.flatten(),
            'error': error.flatten(),
            'anomaly': anomalies.flatten().astype(int)
        })
        
        return result, threshold
    except Exception as e:
        st.error(f"Lỗi khi phát hiện bất thường: {e}")
        return None, None

def create_forecast_plot(result_df, future_result=None, title="Dự báo sản lượng điện"):
    """Tạo biểu đồ dự báo vs thực tế sử dụng Plotly"""
    try:
        fig = go.Figure()
        
        # Thêm đường dự báo
        fig.add_trace(go.Scatter(
            x=result_df['ds'],
            y=result_df['y_pred'],
            mode='lines+markers',
            name='Dự báo (hiện tại)',
            line=dict(color='rgba(65, 105, 225, 0.8)', width=2),
            marker=dict(symbol='circle', size=6)
        ))
        
        # Thêm đường thực tế
        fig.add_trace(go.Scatter(
            x=result_df['ds'],
            y=result_df['y_true'],
            mode='lines+markers',
            name='Thực tế',
            line=dict(color='rgba(50, 205, 50, 0.8)', width=2),
            marker=dict(symbol='circle', size=8)
        ))
        
        # Nếu có dự báo tương lai, thêm vào biểu đồ
        if future_result is not None:
            fig.add_trace(go.Scatter(
                x=future_result['ds'],
                y=future_result['y_pred'],
                mode='lines+markers',
                name='Dự báo tương lai',
                line=dict(color='rgba(255, 165, 0, 0.8)', width=2, dash='dot'),
                marker=dict(symbol='triangle-up', size=8)
            ))
        
        # Nếu có bất thường, thêm điểm đánh dấu
        if 'anomaly' in result_df.columns and result_df['anomaly'].sum() > 0:
            anomalies = result_df[result_df['anomaly'] == 1]
            fig.add_trace(go.Scatter(
                x=anomalies['ds'],
                y=anomalies['y_true'],
                mode='markers',
                name='Bất thường',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='rgba(255, 0, 0, 0.9)',
                    line=dict(width=2, color='rgba(255, 0, 0, 0.9)')
                )
            ))
        
        # Cập nhật layout
        fig.update_layout(
            title=title,
            xaxis_title='Thời gian',
            yaxis_title='Sản lượng điện (kWh)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        fig.update_traces(hovertemplate='%{y:.2f} kWh')
        
        return fig
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ: {e}")
        return None

def create_error_plot(result_df, threshold, title="Sai số dự báo"):
    """Tạo biểu đồ hiển thị sai số dự báo và ngưỡng phát hiện bất thường"""
    try:
        fig = go.Figure()
        
        # Thêm đường sai số
        fig.add_trace(go.Scatter(
            x=result_df['ds'],
            y=result_df['error'],
            mode='lines+markers',
            name='Sai số',
            line=dict(color='rgba(255, 165, 0, 0.8)', width=2),
            marker=dict(symbol='circle', size=6)
        ))
        
        # Thêm đường ngưỡng
        fig.add_trace(go.Scatter(
            x=result_df['ds'],
            y=[threshold] * len(result_df),
            mode='lines',
            name='Ngưỡng phát hiện',
            line=dict(color='rgba(255, 0, 0, 0.6)', width=2, dash='dash')
        ))
        
        # Nếu có bất thường, đánh dấu
        if 'anomaly' in result_df.columns and result_df['anomaly'].sum() > 0:
            anomalies = result_df[result_df['anomaly'] == 1]
            fig.add_trace(go.Scatter(
                x=anomalies['ds'],
                y=anomalies['error'],
                mode='markers',
                name='Bất thường',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='rgba(255, 0, 0, 0.9)',
                    line=dict(width=2, color='rgba(255, 0, 0, 0.9)')
                )
            ))
        
        # Cập nhật layout
        fig.update_layout(
            title=title,
            xaxis_title='Thời gian',
            yaxis_title='Sai số tuyệt đối (kWh)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        fig.update_traces(hovertemplate='%{y:.2f} kWh')
        
        return fig
    except Exception as e:
        st.error(f"Lỗi khi tạo biểu đồ sai số: {e}")
        return None

def calculate_metrics(result_df):
    """Tính toán các chỉ số đánh giá"""
    try:
        # MAE - Mean Absolute Error
        mae = np.mean(result_df['error'])
        
        # MAPE - Mean Absolute Percentage Error
        mape = np.mean(np.abs(result_df['y_true'] - result_df['y_pred']) / (result_df['y_true'] + 1e-8)) * 100
        
        # Số điểm bất thường
        if 'anomaly' in result_df.columns:
            anomaly_count = result_df['anomaly'].sum()
            anomaly_percent = (anomaly_count / len(result_df)) * 100
        else:
            anomaly_count = 0
            anomaly_percent = 0
        
        return {
            'mae': mae,
            'mape': mape,
            'anomaly_count': anomaly_count,
            'anomaly_percent': anomaly_percent
        }
    except Exception as e:
        st.error(f"Lỗi khi tính toán chỉ số: {e}")
        return None

# --------- MAIN APP ----------

def main():
    # Tiêu đề ứng dụng
    st.title("⚡ Electric Power Generation Forecasting and Operational Anomaly Detection")
    st.sidebar.image("image.png", width=300)
    st.sidebar.header("Input Parameters")
    
    model_path = st.sidebar.text_input("Model Path", value="./models/nhits_model/")
    scalers_path = st.sidebar.text_input("Path to Scaler Objects ", value="./models/scalers.pkl")
    data_path = st.sidebar.text_input("Path to Preprocessing Scalers", value="data/test.csv")
    
    # Kiểm tra các đường dẫn
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model directory not found: {model_path}")
        st.stop()
    if not os.path.exists(scalers_path):
        st.sidebar.error(f"Scaler file not found: {scalers_path}")
        st.stop()
    if not os.path.exists(data_path):
        st.sidebar.error(f"Data file not found: {data_path}")  
        st.stop()
    
    # Tải mô hình, scalers và dữ liệu
    with st.sidebar:
        with st.spinner("Loading the model and dataset..."):
            model, scalers = load_model_and_scalers(model_path, scalers_path)
            df = load_data(data_path)
            unique_ids = get_unique_ids(df)
    
    selected_id = st.sidebar.selectbox("Select electricity meter (unique_id)", unique_ids)
    enable_future_forecast = st.sidebar.checkbox("Next day forecast", value=False)
    
    forecast_days = 7  
    if enable_future_forecast:
        forecast_days = st.sidebar.slider("Number of forecast days", 1, 30, 7)
        
        start_forecast_date = st.sidebar.date_input(
            "Select forecast start date",
            datetime.now().date()
        )
        
        start_forecast_hour = st.sidebar.selectbox(
            " Select forecast start time",
            [0, 6, 12, 18],
            index=0  
        )
    
    samples_per_day = 4
    forecast_len = forecast_days * samples_per_day
    anomaly_threshold = st.sidebar.slider(
        " Anomaly detection threshold (number of standard deviations)",
        1.0, 5.0, 3.0, 0.1
    )
    
    analyze_button = st.sidebar.button("Data analysis")
    
    if analyze_button:
        with st.sidebar:
            with st.spinner(f"Analyzing data for the meter {selected_id}..."):
                data = prepare_data_for_prediction(df, selected_id)
                
                if data is None or len(data) < 10:
                    st.error(f"Insufficient data for the meter {selected_id}")
                    st.stop()
                
                data_scaled, scaler = scale_data(data, scalers, selected_id)
                
                if data_scaled is None or scaler is None:
                    st.error(f"Unable to normalize data for the meter {selected_id}")
                    st.stop()
                
                validation_len = 28 
                train_data = data_scaled.iloc[:-validation_len]
                test_data = data_scaled.iloc[-validation_len:]
                
                validation_forecast = make_forecast(model, train_data, validation_len)
                
                if validation_forecast is None:
                    st.error("Unable to generate forecast")
                    st.stop()
                
                merged = pd.merge(test_data, validation_forecast, on=['ds', 'unique_id'], how='inner')
                result_df, threshold = detect_anomalies(merged, merged, scaler, threshold_std=anomaly_threshold)
                
                if result_df is None:
                    st.error("Unable to detect anomalies")
                    st.stop()
                
                future_result = None
                if enable_future_forecast:
                    future_df = prepare_future_data(
                                                    data, 
                                                    selected_id, 
                                                    forecast_days,
                                                    start_date=start_forecast_date,
                                                    start_hour=start_forecast_hour
                                                )
                    
                    future_forecast = make_forecast(model, data_scaled, len(future_df))
                    
                    if future_forecast is not None:
                        min_len = min(len(future_df), len(future_forecast))
                        future_result = pd.DataFrame({
                                            'ds': future_df['ds'].values[:min_len],
                                            'y_pred': scaler.inverse_transform(future_forecast[['NHITS']][:min_len]).flatten()
                                        })
                        print(f"Length of future_df: {len(future_df)}, Length of future_forecast: {len(future_forecast)}")

        metrics = calculate_metrics(result_df)

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{metrics["mae"]:.2f}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">MAE (kWh)</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{metrics["mape"]:.2f}%</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">MAPE</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{metrics["anomaly_count"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Điểm bất thường</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{metrics["anomaly_percent"]:.1f}%</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Tỷ lệ bất thường</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader(f"📈  Forecast results for the meter {selected_id}")
        
        if enable_future_forecast:
            start_datetime_str = f"{start_forecast_date.strftime('%d/%m/%Y')} {start_forecast_hour:02d}:00"
            end_date = start_forecast_date + timedelta(days=forecast_days)
            end_datetime_str = f"{end_date.strftime('%d/%m/%Y')} {start_forecast_hour:02d}:00"
            st.info(f"Forecast from {start_datetime_str} to {end_datetime_str} ({forecast_len} data points) at hours: 00:00, 06:00, 12:00, 18:00")

        forecast_fig = create_forecast_plot(
            result_df, 
            future_result=future_result, 
            title=f"Electricity Production Forecast – Meter {selected_id}"
        )
        if forecast_fig:
            st.plotly_chart(forecast_fig, use_container_width=True)
        
        error_fig = create_error_plot(result_df, threshold, title=f"Forecast Error – Meter {selected_id}")
        if error_fig:
            st.plotly_chart(error_fig, use_container_width=True)
        
        if enable_future_forecast and future_result is not None:
            st.subheader(f"🔮 Forecast for the next {forecast_days} days")
            
            future_display = future_result.copy()
            future_display['ds'] = future_display['ds'].dt.strftime('%d/%m/%Y %H:%M')
            future_display.rename(columns={
                'ds': 'Time',
                'y_pred': 'Forecast (kWh)'
            }, inplace=True)
            
            future_display['Forecast (kWh)'] = future_display['Forecast (kWh)'].round(2)
            
            st.dataframe(
                future_display,
                use_container_width=True,
                hide_index=True
            )
            
            csv_data = future_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download file CSV forecast",
                data=csv_data,
                file_name=f"electricity_forecast_{forecast_days}_days_meter_{selected_id}.csv",
                mime="text/csv",
            )
        
        if metrics["anomaly_count"] > 0:
            st.subheader("🚨  Anomaly point details")
            anomalies = result_df[result_df['anomaly'] == 1].copy()
            anomalies['ds'] = anomalies['ds'].dt.strftime('%d/%m/%Y %H:%M')
            anomalies.rename(columns={
                    'ds': 'Time',
                    'y_true': 'Actual (kWh)',
                    'y_pred': 'Forecast (kWh)',
                    'error': 'Error (kWh)'
                }, inplace=True)
            
            anomalies_display = anomalies[['Time', 'Actual (kWh)', 'Forecast (kWh)', 'Error (kWh)']].copy()
            anomalies_display['Actual (kWh)'] = anomalies_display['Actual (kWh)'].round(2)
            anomalies_display['Forecast (kWh)'] = anomalies_display['Forecast (kWh)'].round(2)
            anomalies_display['Error (kWh)'] = anomalies_display['Error (kWh)'].round(2)
            
            st.dataframe(
                anomalies_display,
                use_container_width=True,
                hide_index=True
            )
            
            csv_data = anomalies_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Anomalies CSV",
                data=csv_data,
                file_name=f"anomalies_meter_{selected_id}.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ No anomalies detected with the current threshold.")

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        ℹ️ **User Guide**  
        1. Select the meter to analyze  
        2. Enable “Forecast for the next day” and choose the number of forecast days  
        3. Adjust the anomaly detection threshold  
        4. Click the “Analyze Data” button  
        """
    )

if __name__ == "__main__":
    main()