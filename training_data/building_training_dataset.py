import numpy as np
import pandas as pd

input_file_path = "cleaned_data/cleaned_backfilled_data.csv"
output_path = "training_data/training_dataset.csv"

# loading the cleaned backfilled data csv


def load_cleaned_data():
    return pd.read_csv(input_file_path)


# preprocessing functions
# handling missing values


def handle_missing_values(df):
    df.replace(
        [np.inf, -np.inf], np.nan, inplace=True
    )  # replaces infinite or invalid numbers
    df.dropna(axis=1, how="all", inplace=True)  # drops completely empty columns (rare)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # filling missing numeric values with the column mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print("Missing values handled.")
    return df


# dropping redundant columns


def drop_redundant_columns(df):
    drop_cols = ["city"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    print("Redundant columns dropped.")
    return df


# setting the target variable


def set_target_variable(df):
    y = df["ow_aqi_index"]
    X = df.drop(columns=["ow_aqi_index"], errors="ignore")
    print("Target variable set: {'ow_aqi_index'}.")
    return X, y


# saving the training dataset


def save_training_dataset(X, y):
    df = X.copy()
    df["ow_aqi_index"] = y
    df.to_csv(output_path, index=False)
    print(f"Training dataset saved to {output_path}.")
    return df


if __name__ == "__main__":
    df = load_cleaned_data()
    df = handle_missing_values(df)
    df = drop_redundant_columns(df)
    X, y = set_target_variable(df)
    training_dataset = save_training_dataset(X, y)
    print("Final dataset shape:", training_dataset.shape)
    print("Columns in the final dataset:", training_dataset.columns.tolist())
    print("Training dataset header:\n", training_dataset.head())
