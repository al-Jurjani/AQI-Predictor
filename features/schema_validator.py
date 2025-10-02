import pandas as pd

EXPECTED_COLUMNS = {
    "timestamp_utc": "datetime64[ns, UTC]",
    "city": "object",

    "temp": "float64",
    "temp_feels_like": "float64",
    "humidity": "float64",
    "pressure": "float64",

    "wind_speed": "float64",
    "wind_deg": "float64",
    "wind_gust": "float64",

    "co": "float64",
    "no": "float64",
    "no2": "float64",
    "o3": "float64",
    "so2": "float64",
    "pm2_5": "float64",
    "pm10": "float64",
    "nh3": "float64",

    "ow_aqi_index": "float64",
}

def validate_schema(df: pd.DataFrame):
    # Ensure all required columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Coerce types
    import numpy as np

    for col, dtype in EXPECTED_COLUMNS.items():
        if col in df.columns:
            try:
                if dtype.startswith("datetime64"):
                    df[col] = pd.to_datetime(df[col], utc=True)
                elif dtype == "float64":
                    df[col] = df[col].astype(np.float64)
                elif dtype == "object":
                    df[col] = df[col].astype(object)
            except Exception:
                raise TypeError(f"Column {col} cannot be converted to {dtype}")
    
    assert df["pm2_5"].notnull().any(), "No PM2.5 values found!"
    assert df["timestamp_utc"].is_monotonic_increasing, "Timestamps not sorted!"

    print("Schema validation passed!")
    return df
