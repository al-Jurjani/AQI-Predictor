import joblib
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# data_file_path = "training_data/training_dataset.csv"
# output_path = "models/baseline_metrics.csv"

# df = pd.read_csv(data_file_path)
# X = df.drop(columns=["ow_aqi_index", "timestamp_utc"], errors="ignore")
# y = df["ow_aqi_index"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
# print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# evaluating model function
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"Now training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Results:")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print("-"*21)
    return {"Model": model, "Model Name": name, "RMSE": rmse, "MAE": mae, "R2": r2}

# results = []

# models = {
#     "Linear Regression": LinearRegression(),
#     "Ridge Regression": Ridge(alpha=1.0),
#     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
# }
# for name, model in models.items():
#     result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
#     results.append(result)

# metrics_df = pd.DataFrame(results)
# # saving as a csv
# metrics_df.to_csv(output_path, index = False)

# turning this into a single function to be used by automated_training.py
def train_and_evaluate_models(data_file_path, output_path, test_size=0.2, split_random_state=21):
    # --- Main Function Logic ---
    print(f"Loading data from: {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
        return None, None, None

    # Prepare data
    X = df.drop(columns=["ow_aqi_index", "timestamp_utc"], errors="ignore")
    y = df["ow_aqi_index"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    # Train and evaluate
    results = []
    best_rmse = np.inf
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        if result["RMSE"] < best_rmse:
            best_rmse = result["RMSE"]
            best_model_name = result["Model Name"]
            best_model = result["Model"]
        results.append(result)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("models/baseline_metrics.csv", index=False)
    print(f"Training model metrics saved at: models/baseline_metrics.csv")

    print(f"Best model of this training batch: {best_model_name}") 
    best_model_path = f"models/{best_model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(best_model, best_model_path)

    # best_model = metrics_df.sort_values(by="RMSE").iloc[0]["Model"]
     
    return best_model,best_model_name, metrics_df

# This block allows the script to be run directly
if __name__ == "__main__":
    
    # Use the original hardcoded paths as defaults when running as a script
    DEFAULT_DATA_FILE = "training_data/training_dataset.csv"
    DEFAULT_OUTPUT_FILE = "models/baseline_metrics.csv"

    print("--- Running Model Training as Standalone Script ---")
    
    # Call the function
    metrics = train_and_evaluate_models(
        data_file_path=DEFAULT_DATA_FILE,
        output_path=DEFAULT_OUTPUT_FILE
    )
    
    if metrics is not None:
        print("\n--- Training Complete ---")
        print("Final Metrics:")
        # print(metrics.to_string())