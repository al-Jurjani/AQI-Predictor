import os

import hopsworks
from dotenv import load_dotenv

# CONFIG
HOPSWORKS_PROJECT_NAME = "jurjanji_AQI"
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("hopsworks_api_key")
FEATURE_GROUP_NAME = "air_quality_data"
FEATURE_GROUP_VERSION = 1

if HOPSWORKS_API_KEY is not None or HOPSWORKS_PROJECT_NAME is not None:
    print("Attempting to load from Hopsworks Feature Store...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT_NAME
    )
    fs = project.get_feature_store()
    try:
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
        )
        df = fg.read()
        print("Loaded from Hopsworks:", df.shape)
    except Exception as e:
        print("Could not read feature group from Hopsworks:", e)
    else:
        print("Hopsworks not available or API key/project not set")

print(df[["timestamp_utc", "ow_aqi_index"]].tail())
