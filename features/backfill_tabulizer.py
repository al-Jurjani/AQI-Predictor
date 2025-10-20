import glob
import pandas as pd
import os
import shutil

from tabularize_raw_data import tabularize_raw_data
from schema_validator import validate_schema 
from time_based_features import add_time_based_features
from derived_features import add_derived_features


def tabularize_all_files():
    all_files = glob.glob("backfilled_data/*.json")
    dfs = [tabularize_raw_data(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True) if dfs else None
    df = df.drop_duplicates(subset=['timestamp_utc', 'city']) if df is not None else None

    if df is not None:
        df = validate_schema(df)
        df = add_time_based_features(df)
        df = add_derived_features(df)
        df = df.reset_index() if df.index.name == "timestamp_utc" else df
        # Make sure dtype is correct
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    for f in all_files:
        try:
            destination = os.path.join("backfilled_data/archive", os.path.basename(f))
            shutil.move(f, destination)
            print(f"📦 Moved {f} → {destination}")
        except Exception as e:
            print(f"⚠️ Could not move {f}: {e}")
    
    return df

# Turn df into csv and store in cleaned_data folder
def save_cleaned_data(df: pd.DataFrame, output_filepath: str):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"💾 Cleaned data saved to {output_filepath}")

if __name__ == "__main__":
    df = tabularize_all_files()
    if df is not None:
        # df = validate_schema(df)

        # df = add_time_based_features(df)
        # df = add_derived_features(df)
        
        print("A preview of the dataframe is as follows: \n")
        print(df.head())
        print("-"*21)
        print("The features of the dataframe include: \n", df.columns)
    
    # Save the cleaned data to a CSV file
    if df is not None:
        output_filepath = "cleaned_data/cleaned_backfilled_data.csv"
        save_cleaned_data(df, output_filepath)
