import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
import os

st.set_page_config(
    page_title="Electric Power Forecasting & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


@st.cache_resource
def load_model():
    nf = NeuralForecast.load(path='model_nhits')
    return nf

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

def predict_single_uid(model, history, uid):
    df_input = history.copy()
    df_input['unique_id'] = uid
    pred = model.predict(df=df_input)
    return pred

def plot_forecast(pred_df, scaler, test_df):
    merged = pd.merge(test_df, pred_df, on=['unique_id', 'ds'], how='inner')
    if merged.empty:
        st.warning("No data available for visualization.")
        return

    y_true = scaler.inverse_transform(merged[['y']])
    y_pred = scaler.inverse_transform(merged[['NHITS']])
    error = np.abs(y_true - y_pred)
    threshold = error.mean() + 3 * error.std()
    anomaly = (error > threshold).flatten()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=merged['ds'],
        y=y_true.flatten(),
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=merged['ds'],
        y=y_pred.flatten(),
        mode='lines+markers',
        name='Forecast',
        line=dict(color='green')
    ))

    if any(anomaly):
        fig.add_trace(go.Scatter(
            x=merged['ds'][anomaly],
            y=y_true[anomaly].flatten(),
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10)
        ))

    fig.update_layout(
        title="Forecast & Operational Anomaly Detection",
        xaxis_title="Timestamp",
        yaxis_title="Energy Consumption (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


train_path = 'data/train.csv'
test_path = 'data/test.csv'
input_len = 56
forecast_len = 28

st.title("⚡ Electric Power Generation Forecasting and Operational Anomaly Detection")
st.sidebar.image("image.png", width=300)
st.sidebar.header("Input Parameters")

train_df, test_df, scalers = load_and_preprocess(train_path, test_path)
model = load_model()

with st.sidebar:
    unique_ids = sorted(test_df['unique_id'].unique())
    uid = st.selectbox("Select Meter (unique_id):", unique_ids)

    start_dates = test_df[test_df['unique_id'] == uid]['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    start_date_str = st.selectbox("Select Forecast Start Time:", start_dates)
    start_date = pd.to_datetime(start_date_str)

    forecast_button = st.button("🔍 Forecast", use_container_width=True)

    st.divider()
    st.markdown("### Information")
    st.info(f"The model uses {input_len} historical data points")
    st.info(f"Forecast length: {forecast_len} steps")

if forecast_button:
    with st.spinner("Running forecast..."):
        history = get_history(train_df, test_df, uid, start_date, input_len)
        if history is None:
            st.error("Insufficient historical data for forecasting.")
        else:
            try:
                pred = predict_single_uid(model, history, uid)
                plot_forecast(pred, scalers[uid], test_df[test_df['unique_id'] == uid])

                with st.expander("Show Forecast Details"):
                    merged = pd.merge(
                        test_df[test_df['unique_id'] == uid],
                        pred,
                        on=['unique_id', 'ds'],
                        how='inner'
                    )

                    if not merged.empty:
                        merged['y_original'] = scalers[uid].inverse_transform(merged[['y']])
                        merged['forecast'] = scalers[uid].inverse_transform(merged[['NHITS']])
                        merged['error'] = np.abs(merged['y_original'] - merged['forecast'])

                        st.dataframe(
                            merged[['ds', 'y_original', 'forecast', 'error']].rename(
                                columns={
                                    'ds': 'Timestamp',
                                    'y_original': 'Actual (kWh)',
                                    'forecast': 'Forecast (kWh)',
                                    'error': 'Error (kWh)'
                                }
                            )
                        )
            except Exception as e:
                st.error(f"Forecasting error: {e}")
else:
    st.info("Please select a meter and forecast start time in the sidebar, then click Forecast.")
