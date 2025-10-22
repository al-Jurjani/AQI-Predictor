import hopsworks
from datetime import datetime
import pandas as pd
from hsfs.feature_group import FeatureGroup
import os
from dotenv import load_dotenv

from tabulizer import tabularize_all_files

load_dotenv()
the_hopsworks_api_key = os.getenv("hopsworks_api_key")
if the_hopsworks_api_key:
    the_project = hopsworks.login(api_key_value=the_hopsworks_api_key)
else:
    the_project = hopsworks.login() # login via console/terminal
the_fs = the_project.get_feature_store()

# if the_fs.get_feature_group("air_quality_data", 1):
#     feature_group = the_fs.get_feature_group("air_quality_data", 1)
# else:
#     feature_group =  the_fs.create_feature_group(
#         name="air_quality_data",
#         version=1,
#         primary_key=["city", "timestamp_utc"],
#         description="Hourly engineered features data for predicting AQI index.",
#         online_enabled=True)

try:
    feature_group = the_fs.get_feature_group("air_quality_data", 1)
    print(f"Feature group name: {feature_group.name}, version: {feature_group.version} found.")
except Exception:
    feature_group = the_fs.create_feature_group(
        name="air_quality_data",
        version=1,
        primary_key=["city", "timestamp_key"],
        event_time = "timestamp_utc",
        description="Hourly engineered features data for predicting AQI index.",
        online_enabled=True
    )
    print(f"Feature group name: {feature_group.name}, version: {feature_group.version} created.")


if feature_group is not None:
    the_df = tabularize_all_files()
    if the_df is not None:
        the_df = the_df.reset_index() if the_df.index.name == "timestamp_utc" else the_df
        if "timestamp_utc" not in the_df.columns:
            raise ValueError("timestamp_utc column missing before insert")

        # Make sure dtype is correct
        the_df["timestamp_utc"] = pd.to_datetime(the_df["timestamp_utc"], utc=True)
        the_df["timestamp_key"] = the_df["timestamp_utc"].astype(str) # for pk of hopsworks fs

        # print("A preview of the dataframe is as follows: \n")
        # print(the_df.head())
        # print("-"*21)
        # print("The features of the dataframe include: \n", the_df.columns)
        # print(the_df.dtypes["timestamp_utc"])
        # print("-"*21)

        # inserting in the feature store
        print("Inserting the dataframe into the feature group...")
        feature_group.insert(the_df, write_options={"wait_for_job": True})
    else:
        print("No data available to insert into the feature group.")
else:
    print("No feature group found.")