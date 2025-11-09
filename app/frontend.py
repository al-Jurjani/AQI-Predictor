import os
from datetime import timedelta

import hopsworks
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="JurjaniX10Pearls AQI Predictor - Karachi",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme colors
bg_color = "#0a0e27"
text_color = "#e8eaf6"
card_bg = "#1a1f3a"
accent_color = "#4fc3f7"
plot_template = "plotly_dark"

st.markdown(
    f"""
    <style>
        .main {{
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
            color: {text_color};
        }}
        .stApp {{
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        }}
        .metric-card {{
            background: linear-gradient(145deg, {card_bg}, #252d4f);
            padding: 28px 20px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
            border: 1px solid rgba(79, 195, 247, 0.1);
            transition: transform 0.2s;
            margin-bottom: 10px;
        }}
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(79, 195, 247, 0.2);
        }}
        .metric-value {{
            font-size: 3rem;
            font-weight: 700;
            color: {text_color};
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
            letter-spacing: -1px;
        }}
        .metric-label {{
            font-size: 0.95rem;
            color: {text_color};
            opacity: 0.75;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 500;
            margin-bottom: 12px;
        }}
        .metric-category {{
            font-size: 1rem;
            margin-top: 12px;
            font-weight: 600;
            opacity: 0.9;
        }}
        h1 {{
            color: {text_color} !important;
            font-weight: 700 !important;
            font-size: 2.8rem !important;
            margin-bottom: 0.3rem !important;
            text-shadow: 0 2px 12px rgba(79, 195, 247, 0.3);
        }}
        h2, h3 {{
            color: {text_color} !important;
            font-weight: 600 !important;
        }}
        .last-updated {{
            color: {accent_color};
            font-size: 1rem;
            font-weight: 500;
            opacity: 0.9;
        }}
        hr {{
            border-color: rgba(79, 195, 247, 0.2) !important;
            margin: 2rem 0 !important;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
        }}
        .stPlotlyChart {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }}
    </style>
""",
    unsafe_allow_html=True,
)

try:
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("hopsworks_api_key")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="jurjanji_AQI")

    st.sidebar.success("✅ Connected to Hopsworks")

    mr = project.get_model_registry()
    fs = project.get_feature_store()

    EVALUATION_METRIC = "rmse"
    SORT_METRICS_BY = "min"
    best_model = mr.get_best_model("best_aqi_model", EVALUATION_METRIC, SORT_METRICS_BY)

    st.sidebar.success(f"✅ Best Model Version: v{best_model.version}")
    st.sidebar.metric("RMSE", f"{best_model.training_metrics.get('rmse', 'N/A')}")
    # st.sidebar.info("Best model loaded!")

    model_dir = best_model.download()
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith("_model.pkl")]

    if not pkl_files:
        st.error("No model pickle file found!")
        st.stop()

    model_path = os.path.join(model_dir, pkl_files[0])
    model = joblib.load(model_path)
    st.sidebar.success(f"✅ Loaded: {pkl_files[0]}")

    fg = fs.get_feature_group(name="air_quality_data", version=1)
    df = fg.read()
    st.sidebar.success(f"✅ Loaded {len(df)} records")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
    st.stop()

# Timezone conversion
PKT = pytz.timezone("Asia/Karachi")

if "timestamp_utc" in df.columns:
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = (
            df["timestamp_utc"].dt.tz_localize("UTC").dt.tz_convert(PKT)
        )
    else:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert(PKT)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    time_col = "timestamp_utc"
elif "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(PKT)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(PKT)
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_col = "timestamp"
else:
    st.error("No timestamp column found!")
    st.stop()

exclude_cols = [
    "ow_aqi_index",
    "timestamp_utc",
    "city",
    "timestamp_key",
    "timestamp",
    time_col,
    "temp_feels_like",
    "co_no2_ratio",
    "temp_rolling_avg_4h",
    "temp_rolling_avg_30d",
    "temp_rolling_avg_7d",
    "temp_rolling_avg_24h",
    "so2_no2_ratio",
    "temp",
    "so2_wind_disp",
    "co_wind_disp",
    "nh3_wind_disp",
    "pm2_5_wind_disp",
    "no2_wind_disp",
    "no_wind_disp",
    "pm10_wind_disp",
    "o3_wind_disp",
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
aqi_col = "ow_aqi_index" if "ow_aqi_index" in df.columns else "AQI"


def predict_current(model, df, feature_cols):
    latest_row = df.iloc[-1:][feature_cols]
    return model.predict(latest_row)[0]


def forecast_future(model, df, feature_cols, hours=[24, 48, 72]):
    forecasts = {}
    current_features = df.iloc[-1][feature_cols].copy()
    latest_timestamp = df[time_col].iloc[-1]
    previous_aqi = df["ow_aqi_index"].iloc[-1]

    for h in hours:
        future_features = current_features.copy()
        future_time = latest_timestamp + timedelta(hours=h)

        future_features["hour"] = future_time.hour
        future_features["day_of_week"] = future_time.dayofweek
        future_features["day_of_month"] = future_time.day
        future_features["month"] = future_time.month
        future_features["hour_sin"] = np.sin(2 * np.pi * future_time.hour / 24)
        future_features["hour_cos"] = np.cos(2 * np.pi * future_time.hour / 24)
        future_features["day_of_week_sin"] = np.sin(
            2 * np.pi * future_time.dayofweek / 7
        )
        future_features["day_of_week_cos"] = np.cos(
            2 * np.pi * future_time.dayofweek / 7
        )
        future_features["month_sin"] = np.sin(2 * np.pi * (future_time.month - 1) / 12)
        future_features["month_cos"] = np.cos(2 * np.pi * (future_time.month - 1) / 12)

        if "aqi_delta_3h" in future_features.index:
            future_features["aqi_delta_3h"] = previous_aqi - df["ow_aqi_index"].iloc[-2]
        if "aqi_delta_24h" in future_features.index:
            future_features["aqi_delta_24h"] = (
                previous_aqi - df["ow_aqi_index"].iloc[-24] if len(df) > 24 else 0
            )
        if "aqi_pct_change_3h" in future_features.index:
            old_val = df["ow_aqi_index"].iloc[-2]
            future_features["aqi_pct_change_3h"] = (
                ((previous_aqi - old_val) / old_val * 100) if old_val != 0 else 0
            )
        if "aqi_pct_change_24h" in future_features.index:
            old_val = df["ow_aqi_index"].iloc[-24] if len(df) > 24 else previous_aqi
            future_features["aqi_pct_change_24h"] = (
                ((previous_aqi - old_val) / old_val * 100) if old_val != 0 else 0
            )

        pred = model.predict(future_features.values.reshape(1, -1))[0]
        forecasts[h] = pred
        previous_aqi = pred
        current_features = future_features

    return forecasts


current_aqi = predict_current(model, df, feature_cols)
future_forecasts = forecast_future(model, df, feature_cols)


def get_aqi_category(aqi):
    if aqi <= 1:
        return "Good", "#00e400"
    elif aqi <= 2:
        return "Moderate", "#ffff00"
    elif aqi <= 3:
        return "Unhealthy for Sensitive", "#ff7e00"
    elif aqi <= 4:
        return "Unhealthy", "#ff0000"
    else:
        return "Hazardous", "#8b0000"


# Header
st.title("Karachi AQI Predictor")
st.markdown(
    "Made by Zuhair Farhan - Github repo link: https://github.com/al-Jurjani/AQI-Predictor"
)
st.markdown(
    f'<p class="last-updated">Last updated: {df[time_col].iloc[-1].strftime("%Y-%m-%d %H:%M:%S PKT")}</p>',
    unsafe_allow_html=True,
)
st.markdown("")

# Predictions
st.subheader("AQI Predictions")

col1, col2, col3, col4 = st.columns(4)

predictions = [
    (col1, "Current AQI", current_aqi),
    (col2, "+24 Hours", future_forecasts[24]),
    (col3, "+48 Hours", future_forecasts[48]),
    (col4, "+72 Hours", future_forecasts[72]),
]

for col, label, value in predictions:
    category, color = get_aqi_category(value)
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color: {color};">{value:.2f}</div>
                <div class="metric-category" style="color: {color};">{category}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Trend Chart
st.subheader("7-Day AQI Trend")

past_week = df.tail(7 * 24)

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=past_week[time_col],
        y=past_week[aqi_col],
        mode="lines",
        name="Historical AQI",
        line=dict(color="#4fc3f7", width=3),
        fill="tozeroy",
        fillcolor="rgba(79, 195, 247, 0.1)",
    )
)

fig1.add_hline(
    y=current_aqi,
    line_dash="dash",
    line_color="#ffa726",
    line_width=2,
    annotation_text=f"Current: {current_aqi:.2f}",
    annotation_position="top right",
)

fig1.update_layout(
    template=plot_template,
    xaxis_title="",
    yaxis_title="AQI Index",
    hovermode="x unified",
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    margin=dict(l=50, r=30, t=30, b=50),
)

st.plotly_chart(fig1, use_container_width=True)

# Rolling Averages
st.subheader("Rolling AQI Averages")

df["AQI_24h_avg"] = df[aqi_col].rolling(window=24, min_periods=1).mean()
df["AQI_72h_avg"] = df[aqi_col].rolling(window=72, min_periods=1).mean()

recent_data = df.tail(7 * 24)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=recent_data[time_col],
        y=recent_data[aqi_col],
        mode="lines",
        name="Actual AQI",
        line=dict(color="rgba(255,255,255,0.2)", width=1),
    )
)
fig2.add_trace(
    go.Scatter(
        x=recent_data[time_col],
        y=recent_data["AQI_24h_avg"],
        mode="lines",
        name="24h Average",
        line=dict(color="#ff7043", width=3),
    )
)
fig2.add_trace(
    go.Scatter(
        x=recent_data[time_col],
        y=recent_data["AQI_72h_avg"],
        mode="lines",
        name="72h Average",
        line=dict(color="#66bb6a", width=3),
    )
)

fig2.update_layout(
    template=plot_template,
    xaxis_title="",
    yaxis_title="AQI Index",
    hovermode="x unified",
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    margin=dict(l=50, r=30, t=30, b=50),
)

st.plotly_chart(fig2, use_container_width=True)


st.markdown("---")
st.subheader("Understanding AQI Scales Across Regions")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    **OpenWeather Scale** (This Dashboard)
    - **1** = Good 🟢
    - **2** = Fair 🟡
    - **3** = Moderate 🟠
    - **4** = Poor 🔴
    - **5** = Very Poor ⚫

    *Decimal values (e.g., 2.3) show gradations between levels*
    """
    )

    st.markdown(
        """
    **Europe (EAQI) Scale**
    - 0-25 = Very Low (≈ OW 1)
    - 25-50 = Low (≈ OW 2)
    - 50-75 = Medium (≈ OW 3)
    - 75-100 = High (≈ OW 4)
    - 100+ = Very High (≈ OW 5)
    """
    )

with col2:
    st.markdown(
        """
    **USA (EPA) Scale**
    - 0-50 = Good (≈ OW 1)
    - 51-100 = Moderate (≈ OW 2)
    - 101-150 = Unhealthy for Sensitive (≈ OW 3)
    - 151-200 = Unhealthy (≈ OW 4)
    - 201-300 = Very Unhealthy (≈ OW 5)
    - 301+ = Hazardous
    """
    )

    st.markdown(
        """
    **China Scale**
    - 0-50 = Excellent (≈ OW 1)
    - 51-100 = Good (≈ OW 1-2)
    - 101-150 = Lightly Polluted (≈ OW 3)
    - 151-200 = Moderately Polluted (≈ OW 3-4)
    - 201-300 = Heavily Polluted (≈ OW 4-5)
    - 300+ = Severely Polluted
    """
    )

st.info(
    "**Quick Reference:** An OpenWeather prediction of **2.3** (Fair) ≈ **60-80** on USA EPA scale, and ≈ **30-40** on Europe scale, and ≈ **70-90** on China scale."
)
st.info(
    "**Sources:** https://openweathermap.org/api/air-pollution and https://openweathermap.org/air-pollution-index-levels"
)

# Footer
st.markdown("---")
st.caption("Data: Hopsworks Feature Store | Auto-updated daily | JurjaniX10Pearls")

with st.sidebar:
    st.markdown("---")
    st.markdown("### AQI Scale")
    st.markdown(
        """
    🟢 **0-1** Good
    🟡 **1-2** Moderate
    🟠 **2-3** Unhealthy (Sensitive)
    🔴 **3-4** Unhealthy
    ⚫ **4-5** Hazardous
    """
    )
