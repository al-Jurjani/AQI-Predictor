import numpy as np
import pandas as pd


def add_rolling_avgs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")  # optional but cleaner
    df = df.sort_values("timestamp_utc")

    for col in [
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "temp",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_deg",
        "wind_gust",
    ]:
        df[f"{col}_rolling_avg_4h"] = df[col].rolling("4h").mean()
        df[f"{col}_rolling_avg_24h"] = df[col].rolling("24h").mean()
        df[f"{col}_rolling_avg_7d"] = df[col].rolling("7d").mean()
        df[f"{col}_rolling_avg_30d"] = df[col].rolling("30d").mean()
    return df


def add_aqi_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp_utc")

    df["aqi_delta_3h"] = df["ow_aqi_index"].diff(periods=3)
    df["aqi_delta_24h"] = df["ow_aqi_index"].diff(periods=24)
    df["aqi_pct_change_3h"] = df["ow_aqi_index"].pct_change(periods=3)
    df["aqi_pct_change_24h"] = df["ow_aqi_index"].pct_change(periods=24)
    return df


def add_pollutant_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)
    df["no2_o3_ratio"] = df["no2"] / (df["o3"] + 1e-6)
    df["so2_no2_ratio"] = df["so2"] / (df["no2"] + 1e-6)
    df["co_no2_ratio"] = df["co"] / (df["no2"] + 1e-6)

    return df


def add_wind_dispersion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]:
        df[f"{col}_wind_disp"] = df[col] / (df["wind_speed"] + 1e-6)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts hour/day/month into cyclical (sin/cos) for ML models."""
    df = df.copy()
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "day_of_week" in df.columns:
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# main function to add all derived features


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rolling_avgs(df)
    df = add_aqi_deltas(df)
    df = add_pollutant_ratios(df)
    df = add_wind_dispersion_features(df)
    df = add_cyclical_time_features(df)
    return df
