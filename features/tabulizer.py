import glob
import pandas as pd

from tabularize_raw_data import tabularize_raw_data
from schema_validator import validate_schema 
from time_based_features import add_time_based_features
from derived_features import add_derived_features


def tabularize_all_files():
    all_files = glob.glob("raw_data/*.json")
    dfs = [tabularize_raw_data(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True) if dfs else None
    df = df.drop_duplicates(subset=['timestamp_utc', 'city']) if df is not None else None
    return df

if __name__ == "__main__":
    df = tabularize_all_files()
    if df is not None:
        df = validate_schema(df)

        df = add_time_based_features(df)
        df = add_derived_features(df)
        
        print("A preview of the dataframe is as follows: \n")
        print(df.head())
        print("-"*21)
        print("The features of the dataframe include: \n", df.columns)