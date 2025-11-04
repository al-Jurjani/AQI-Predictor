import os
import sys
from typing import Optional

import hopsworks
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.derived_features import add_derived_features
from features.schema_validator import validate_schema
from features.time_based_features import add_time_based_features

load_dotenv()
the_hopsworks_api_key = os.getenv("hopsworks_api_key")

WEATHER_CSV_PATH = "fetch_data/karachi_complete_weather_data.csv"  # Open-Meteo CSV
POLLUTION_CSV_PATH = "fetch_data/karachi_complete_air_quality_data.xlsx"  # Kaggle CSV

# Basic meta
CITY = "Karachi"
LAT = 24.9056
LON = 67.0822

# Options
CONVERT_TEMP_TO_KELVIN = (
    True  # set True if you want temps in Kelvin (example JSON looked Kelvin)
)


# AQI mapping helper (if kaggle already uses 1-5, this returns same)
def map_main_aqi_to_ow_scale(aqi_value) -> Optional[int]:
    if aqi_value is None:
        return None
    try:
        aqi = int(aqi_value)
        if 1 <= aqi <= 5:
            return aqi

        if 0 <= aqi <= 500:
            if aqi <= 50:
                return 1
            elif aqi <= 100:
                return 2
            elif aqi <= 200:
                return 3
            elif aqi <= 300:
                return 4
            else:
                return 5
    except Exception:
        return None


def load_weather_csv(path: str) -> pd.DataFrame:
    print(f"Loading weather CSV: {path}")
    df = pd.read_csv(path, parse_dates=["time"])

    # normalize column names to common names
    df.rename(
        columns={
            "time": "timestamp_utc",
            "temperature_2m": "temp",
            "apparent_temperature": "feels_like",
            "surface_pressure": "pressure",
            "relative_humidity_2m": "humidity",
            "pressure_msl": "pressure_msl",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "wind_gusts_10m": "wind_gusts",
        },
        inplace=True,
    )

    # ensure timezone awareness: assume times are UTC if naive
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    else:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("UTC")

    return df


def load_pollution_csv(path: str) -> pd.DataFrame:
    print(f"Loading pollution CSV: {path}")
    df = pd.read_excel(path, parse_dates=["datetime"])

    df = df.rename(columns={"datetime": "timestamp_utc"})
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    else:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("UTC")

    # normalize pollutant columns names if different
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if "main.aqi" == low:
            rename_map[col] = "main_aqi"
        elif "components.co" == low:
            rename_map[col] = "co"
        elif "components.no" == low:
            rename_map[col] = "no"
        elif "components.no2" == low:
            rename_map[col] = "no2"
        elif "components.o3" == low:
            rename_map[col] = "o3"
        elif "components.so2" == low:
            rename_map[col] = "so2"
        elif "components.pm2_5" == low:
            rename_map[col] = "pm2_5"
        elif "components.pm10" == low:
            rename_map[col] = "pm10"
        elif "components.nh3" == low:
            rename_map[col] = "nh3"
    df.rename(columns=rename_map, inplace=True)
    return df


def merge_weather_pollution(df_weather: pd.DataFrame, df_poll: pd.DataFrame):
    df_merged = pd.merge(
        df_weather, df_poll, on="timestamp_utc", how="inner", suffixes=("_w", "_p")
    )
    return df_merged


if __name__ == "__main__":
    weather_df = load_weather_csv(WEATHER_CSV_PATH)
    pollution_df = load_pollution_csv(POLLUTION_CSV_PATH)

    print("First five Weather DF Samples: \n", weather_df.head())
    print("\n Last five Weather DF Samples: \n", weather_df.tail())
    print("---" * 40)
    print("First five Pollution DF Samples: \n", pollution_df.head())
    print("\n Last five Pollution DF Samples: \n", pollution_df.tail())
    print("---" * 40)
    print("Weather Df Columns:", weather_df.columns.tolist())
    print("Pollution Df Columns:", pollution_df.columns.tolist())
    print("---" * 40)
    print("Weather DF shape:", weather_df.shape)
    print("Pollution DF shape:", pollution_df.shape)
    print("---" * 40)
    print("MERGING DATAFRAMES...")
    merged_df = merge_weather_pollution(weather_df, pollution_df)
    print("Merged DF shape:", merged_df.shape)
    print("Merged DF head:\n", merged_df.head())
    print("Merged DF tail:\n", merged_df.tail())
    print("---" * 40)
    print("Merged DF columns: ", merged_df.columns.tolist())

    drop_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "shortwave_radiation",
    ]
    merged_df = merged_df.drop(columns=drop_columns)

    merged_df["city"] = "Karachi"
    merged_df = merged_df.rename(
        columns={
            "feels_like": "temp_feels_like",
            "wind_direction": "wind_deg",
            "wind_gusts": "wind_gust",
            "main_aqi": "ow_aqi_index",
        }
    )
    print("Merged DF columns: ", merged_df.columns.tolist())
    print("---" * 40)
    print("---" * 40)
    print("---" * 40)
    print("---" * 40)
    print("---" * 40)
    print("Preparing to upload to Hopsworks-....................")

    merged_df["timestamp_utc"] = pd.to_datetime(merged_df["timestamp_utc"], utc=True)
    expected_cols = [
        "timestamp_utc",
        "city",
        "temp",
        "temp_feels_like",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_deg",
        "wind_gust",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "ow_aqi_index",
    ]
    merged_df = merged_df[expected_cols]

    df_valid = validate_schema(merged_df)
    df_features = df_valid.pipe(add_time_based_features).pipe(add_derived_features)

    df_features = (
        df_features.reset_index()
        if df_features.index.name == "timestamp_utc"
        else df_features
    )

    print("Final DF shape: ", df_features.shape)
    print("Final DF columns: ", df_features.columns.tolist())
    print("Final DF Head Samples: \n", df_features.head())
    print("---" * 40)
    print("Final DF Tail Samples: \n", df_features.tail())
    print("---" * 40)
    print("---" * 40)
    print("---" * 40)
    print("---" * 40)
    print("---" * 40)

    # Uploading to hopsworks feature store
    print("Uploading the backfill data to hopsworks!")
    # the_hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")
    if not the_hopsworks_api_key:
        raise ValueError("the_hopsworks_api_key environment variable not set or empty!")
    if the_hopsworks_api_key:
        the_project = hopsworks.login(api_key_value=the_hopsworks_api_key)
    else:
        the_project = hopsworks.login()  # login via console/terminal
    the_fs = the_project.get_feature_store()

    try:
        feature_group = the_fs.get_feature_group("air_quality_data", 1)
        print(
            f"Feature group name: {feature_group.name}, version: {feature_group.version} found."
        )
    except Exception:
        feature_group = the_fs.create_feature_group(
            name="air_quality_data",
            version=1,
            primary_key=["city", "timestamp_key"],
            event_time="timestamp_utc",
            description="Hourly engineered features data for predicting AQI index.",
            online_enabled=True,
        )
        print(
            f"Feature group name: {feature_group.name}, version: {feature_group.version} created."
        )

    if feature_group is not None:
        # Make sure dtype is correct
        df_features["timestamp_utc"] = pd.to_datetime(
            df_features["timestamp_utc"], utc=True
        )
        df_features["timestamp_key"] = df_features["timestamp_utc"].astype(
            str
        )  # for pk of hopsworks fs

        # inserting in the feature store
        print("Inserting the dataframe into the feature group...")
        feature_group.insert(df_features, write_options={"wait_for_job": True})
    else:
        print("No data available to insert into the feature group.")
        exit()
    print("No feature group found.")
