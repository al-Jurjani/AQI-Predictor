import glob
import pandas as pd
from time_based_features import add_time_based_features
from tabularize_raw_data import tabularize_raw_data

def tabularize_all_files():
    all_files = glob.glob("raw_data/*.json")
    dfs = [tabularize_raw_data(f) for f in all_files]
    df = pd.concat(dfs, ignore_index=True) if dfs else None
    df = df.drop_duplicates(subset=['timestamp_utc', 'city']) if df is not None else None
    return df

if __name__ == "__main__":
    df = tabularize_all_files()
    if df is not None:
        df = add_time_based_features(df)
        print(df.head())
        print("-"*21)
        print("The features of the dataframe include: \n", df.columns)