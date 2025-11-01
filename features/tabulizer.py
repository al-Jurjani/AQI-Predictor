import json
import os

import pandas as pd
from azure.storage.blob import BlobServiceClient
from derived_features import add_derived_features
from dotenv import load_dotenv
from schema_validator import validate_schema
from tabularize_raw_data import tabularize_raw_data
from time_based_features import add_time_based_features

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "aqi-data"


def tabularize_all_files():
    container_name = "aqi-data"
    container_client = blob_service_client.get_container_client(container_name)
    prefix = "raw_data/"

    # A.) list all unarchived blobs
    all_blobs = [
        blob.name
        for blob in container_client.list_blobs(name_starts_with=prefix)
        if "archive/" not in blob.name
    ]

    if not all_blobs:
        print("ℹ️ No new raw JSON files found in blob storage.")
        return None

    dfs = []
    for blob_name in all_blobs:
        try:
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            data = json.loads(blob_client.download_blob().readall())
            df = tabularize_raw_data(
                data
            )  # assume your tabularize_raw_data() accepts dicts
            dfs.append(df)
            print(f"✅ Processed {blob_name}")
        except Exception as e:
            print(f"⚠️ Could not process {blob_name}: {e}")

    # all_files = glob.glob("raw_data/*.json")
    # dfs = [tabularize_raw_data(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True) if dfs else None
    df = (
        df.drop_duplicates(subset=["timestamp_utc", "city"]) if df is not None else None
    )

    if df is not None:
        df = validate_schema(df)
        df = add_time_based_features(df)
        df = add_derived_features(df)

    # for f in all_files:
    #     try:
    #         destination = os.path.join("raw_data/archive", os.path.basename(f))
    #         shutil.move(f, destination)
    #         print(f"📦 Moved {f} → {destination}")
    #     except Exception as e:
    #         print(f"⚠️ Could not move {f}: {e}")

    # B.) move processed blobs to archive
    for blob_name in all_blobs:
        try:
            archive_name = blob_name.replace("raw_data/", "raw_data/archive/")
            src_blob = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )
            dest_blob = blob_service_client.get_blob_client(
                container=container_name, blob=archive_name
            )

            dest_blob.start_copy_from_url(src_blob.url)
            src_blob.delete_blob()
            print(f"📦 Moved {blob_name} → {archive_name}")
        except Exception as e:
            print(f"⚠️ Could not move {blob_name}: {e}")
    return df


if __name__ == "__main__":
    df = tabularize_all_files()
    if df is not None:
        # df = validate_schema(df)
        # df = add_time_based_features(df)
        # df = add_derived_features(df)

        print("A preview of the dataframe is as follows: \n")
        print(df.head())
        print("-" * 21)
        print("The features of the dataframe include: \n", df.columns)
