import pandas as pd


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day_of_week"] = df["timestamp_utc"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["day_of_month"] = df["timestamp_utc"].dt.day
    df["month"] = df["timestamp_utc"].dt.month
    return df
