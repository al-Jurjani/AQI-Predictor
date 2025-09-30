import glob
import pandas as pd
from tabularize_raw_data import tabularize_raw_data

def tabularize_all_files():
    all_files = glob.glob("raw_data/*.json")
    dfs = [tabularize_raw_data(f) for f in all_files]
    return pd.concat(dfs, ignore_index=True) if dfs else None

if __name__ == "__main__":
    df = tabularize_all_files()
    if df is not None:
        print(df.head())