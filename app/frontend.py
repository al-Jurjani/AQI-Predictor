import os

import hopsworks
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# THEME CONFIGURATION
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    st.title("⚙️ Settings")
    theme_toggle = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
    )
    st.session_state.theme = theme_toggle

# Apply theme
if st.session_state.theme == "Dark":
    bg_color = "#0e1117"
    text_color = "#fafafa"
    card_bg = "#262730"
    plot_template = "plotly_dark"
else:
    bg_color = "#ffffff"
    text_color = "#31333F"
    card_bg = "#f0f2f6"
    plot_template = "plotly_white"

st.markdown(
    f"""
    <style>
        .main {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .metric-card {{
            background-color: {card_bg};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: {text_color};
        }}
        .metric-label {{
            font-size: 1rem;
            color: {text_color};
            opacity: 0.7;
        }}
        h1, h2, h3 {{
            color: {text_color} !important;
        }}
    </style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# HOPSWORKS CONNECTION
# -----------------------------
st.sidebar.info("🔄 Loading data...")

try:
    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("hopsworks_api_key")

    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project="jurjanji_AQI")

    st.sidebar.success("✅ Connected to Hopsworks")

    # Get model registry and feature store
    mr = project.get_model_registry()
    fs = project.get_feature_store()

    # Load all model versions
    st.sidebar.info("📦 Loading best model...")

    EVALUATION_METRIC = "rmse"  # or r2, rmse
    SORT_METRICS_BY = "min"  # your sorting criteria
    best_model = mr.get_best_model("best_aqi_model", EVALUATION_METRIC, SORT_METRICS_BY)

    st.sidebar.success(f"✅ Best Model: v{best_model.version}")
    st.sidebar.metric("RMSE", f"{best_model.training_metrics.get('rmse', 'N/A')}")
    st.sidebar.metric("R²", f"{best_model.training_metrics.get('r2', 'N/A')}")

    st.sidebar.info("📦 Best model loaded!")

    # Download model
    st.sidebar.info("⬇️ Downloading model...")
    model_dir = best_model.download()

    # Find pickle file
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith("_model.pkl")]
    if not pkl_files:
        st.error("No model pickle file found in downloaded directory!")
        st.stop()

    model_path = os.path.join(model_dir, pkl_files[0])
    model = joblib.load(model_path)

    st.sidebar.success(f"✅ Loaded: {pkl_files[0]}")

    # Load feature data
    st.sidebar.info("📊 Loading feature data...")
    fg = fs.get_feature_group(name="air_quality_data", version=1)
    df = fg.read()

    st.sidebar.success(f"✅ Loaded {len(df)} records")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
    st.stop()

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
# Sort by timestamp
if "timestamp_utc" in df.columns:
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    time_col = "timestamp_utc"
elif "timestamp" in df.columns:
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_col = "timestamp"
else:
    st.error("No timestamp column found!")
    st.stop()

# Convert to datetime if needed
df[time_col] = pd.to_datetime(df[time_col])

# Identify feature columns
# exclude_cols = [time_col, 'ow_aqi_index', 'aqi_delta_24h', 'aqi_delta_3h',
#                 'aqi_pct_change_24h', 'aqi_pct_change_3h']
exclude_cols = [
    "ow_aqi_index",
    "city",
    "timestamp",
    "timestamp_utc",
    time_col,
    "timestamp_key",
]
# feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Get actual AQI column name
aqi_col = "ow_aqi_index" if "ow_aqi_index" in df.columns else "AQI"


# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def predict_current(model, df, feature_cols):
    """Predict current AQI"""
    latest_row = df.iloc[-1:][feature_cols]
    return model.predict(latest_row)[0]


def forecast_future(model, df, feature_cols, hours=[24, 48, 72]):
    """Forecast future AQI values"""
    forecasts = {}
    latest_features = df.iloc[-1:][feature_cols].copy()

    for h in hours:
        # Simple approach: use latest features for prediction
        pred = model.predict(latest_features)[0]
        forecasts[h] = pred

    return forecasts


# Make predictions
current_aqi = predict_current(model, df, feature_cols)
future_forecasts = forecast_future(model, df, feature_cols)


# -----------------------------
# AQI CATEGORY FUNCTION
# -----------------------------
def get_aqi_category(aqi):
    """Return AQI category and color"""
    if aqi <= 1:
        return "Good", "#00e400"
    elif aqi <= 2:
        return "Moderate", "#ffff00"
    elif aqi <= 3:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 4:
        return "Unhealthy", "#ff0000"
    else:
        return "Hazardous", "#7e0023"


# -----------------------------
# HEADER
# -----------------------------
st.title("🌤️ Pearls AQI Predictor Dashboard")
st.markdown(f"*Last updated: {df[time_col].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}*")

# -----------------------------
# CURRENT & FORECAST AQI METRICS
# -----------------------------
st.subheader("📊 AQI Predictions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    category, color = get_aqi_category(current_aqi)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Current AQI</div>
            <div class="metric-value" style="color: {color};">{current_aqi:.0f}</div>
            <div class="metric-label">{category}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    category, color = get_aqi_category(future_forecasts[24])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Tomorrow (+24h)</div>
            <div class="metric-value" style="color: {color};">{future_forecasts[24]:.0f}</div>
            <div class="metric-label">{category}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    category, color = get_aqi_category(future_forecasts[48])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Day 2 (+48h)</div>
            <div class="metric-value" style="color: {color};">{future_forecasts[48]:.0f}</div>
            <div class="metric-label">{category}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    category, color = get_aqi_category(future_forecasts[72])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Day 3 (+72h)</div>
            <div class="metric-value" style="color: {color};">{future_forecasts[72]:.0f}</div>
            <div class="metric-label">{category}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# -----------------------------
# VISUALIZATIONS
# -----------------------------

# 1. AQI TREND (PAST 7 DAYS)
st.subheader("📈 AQI Trend - Past 7 Days")

past_week = df.tail(7 * 24)  # Last 7 days if hourly data

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=past_week[time_col],
        y=past_week[aqi_col],
        mode="lines",
        name="Historical AQI",
        line=dict(color="#1f77b4", width=2),
    )
)

# Add current prediction line
fig1.add_hline(
    y=current_aqi,
    line_dash="dash",
    line_color="orange",
    annotation_text=f"Current Prediction: {current_aqi:.0f}",
    annotation_position="top right",
)

fig1.update_layout(
    template=plot_template,
    xaxis_title="Date",
    yaxis_title="AQI",
    hovermode="x unified",
    height=400,
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# 2. FORECAST BAR CHART
st.subheader("🔮 3-Day Forecast Comparison")

forecast_df = pd.DataFrame(
    {
        "Period": ["Current", "Day 1", "Day 2", "Day 3"],
        "AQI": [
            current_aqi,
            future_forecasts[24],
            future_forecasts[48],
            future_forecasts[72],
        ],
        "Color": [
            get_aqi_category(v)[1]
            for v in [
                current_aqi,
                future_forecasts[24],
                future_forecasts[48],
                future_forecasts[72],
            ]
        ],
    }
)

fig2 = go.Figure(
    data=[
        go.Bar(
            x=forecast_df["Period"],
            y=forecast_df["AQI"],
            marker_color=forecast_df["Color"],
            text=forecast_df["AQI"].round(0),
            textposition="outside",
        )
    ]
)

fig2.update_layout(
    template=plot_template,
    yaxis_title="AQI Value",
    xaxis_title="Time Period",
    height=400,
    showlegend=False,
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3. FEATURE IMPORTANCE / CORRELATIONS
st.subheader("🔍 Top Factors Influencing AQI")

# Calculate correlations with AQI
correlations = (
    df[feature_cols + [aqi_col]]
    .corr()[aqi_col]
    .drop(aqi_col)
    .sort_values(ascending=False)
)
top_10_corr = correlations.head(10)

fig3 = go.Figure(
    data=[
        go.Bar(
            y=top_10_corr.index,
            x=top_10_corr.values,
            orientation="h",
            marker=dict(
                color=top_10_corr.values, colorscale="RdYlGn", reversescale=True
            ),
        )
    ]
)

fig3.update_layout(
    template=plot_template,
    xaxis_title="Correlation with AQI",
    yaxis_title="Feature",
    height=400,
    showlegend=False,
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# 4. ROLLING AVERAGES
st.subheader("🌀 AQI Rolling Averages")

df["AQI_24h_avg"] = df[aqi_col].rolling(window=24, min_periods=1).mean()
df["AQI_72h_avg"] = df[aqi_col].rolling(window=72, min_periods=1).mean()

recent_data = df.tail(7 * 24)

fig4 = go.Figure()
fig4.add_trace(
    go.Scatter(
        x=recent_data[time_col],
        y=recent_data[aqi_col],
        mode="lines",
        name="Actual AQI",
        line=dict(color="lightgray", width=1),
        opacity=0.5,
    )
)
fig4.add_trace(
    go.Scatter(
        x=recent_data[time_col],
        y=recent_data["AQI_24h_avg"],
        mode="lines",
        name="24h Average",
        line=dict(color="#ff7f0e", width=2),
    )
)
fig4.add_trace(
    go.Scatter(
        x=recent_data[time_col],
        y=recent_data["AQI_72h_avg"],
        mode="lines",
        name="72h Average",
        line=dict(color="#2ca02c", width=2),
    )
)

fig4.update_layout(
    template=plot_template,
    xaxis_title="Date",
    yaxis_title="AQI",
    hovermode="x unified",
    height=400,
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "📡 Data source: Hopsworks Feature Store | 🤖 Model auto-updated daily | 🔄 Predictions refresh on page reload"
)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 AQI Categories")
    st.markdown(
        """
    - **0-1**: Good 🟢
    - **1-2**: Moderate 🟡
    - **2-3**: Unhealthy for Sensitive Groups 🟠
    - **3-4**: Unhealthy 🔴
    - **4-5**: Hazardous ⚫
    """
    )
