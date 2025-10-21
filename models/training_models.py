import datetime
import json
import joblib
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from shap_analysis import generate_shap_analysis

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
def train_and_evaluate_models(data_file_path, test_size=0.2, split_random_state=21):
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
        "Ridge Regression (alpha = 0.3)":Ridge(alpha=0.3),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Random Forest (Altered)": RandomForestRegressor(n_estimators=50, max_depth=2, random_state=42)
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

    # creating a  timestamp 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"models/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    metrics_df = pd.DataFrame(results)
    metrics_csv_path = f"{run_dir}/baseline_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training model metrics saved at: {run_dir}]/baseline_metrics.csv")

    print(f"Best model of this training batch: {best_model_name}") 
    best_model_path = f"{run_dir}/{best_model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"Best model file saved at: {best_model_path}")

    metadata = {
        "best_model": best_model_name,
        "artifact_path": best_model_path,
        "timestamp": timestamp,
        "metrics": {
            "RMSE": float(best_rmse),
            "MAE": float(metrics_df.loc[metrics_df['Model Name'] == best_model_name, 'MAE'].values[0]),
            "R2": float(metrics_df.loc[metrics_df['Model Name'] == best_model_name, 'R2'].values[0])
        },
        "data_source": data_file_path
    }
    metadata_path = os.path.join(run_dir, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Best model meta data saved at: {metadata_path}")

    #SHAP Analysis
    summary_path, bar_path = generate_shap_analysis(best_model, X_train, run_dir)
    if summary_path and bar_path:
        metadata["shap_summary_plot"] = summary_path
        metadata["shap_bar_plot"] = bar_path
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # best_model = metrics_df.sort_values(by="RMSE").iloc[0]["Model"]
     
    return best_model, best_model_name, metrics_df

# This block allows the script to be run directly
if __name__ == "__main__":
    
    # Use the original hardcoded paths as defaults when running as a script
    DEFAULT_DATA_FILE = "training_data/training_dataset.csv"
    DEFAULT_OUTPUT_FILE = "models/baseline_metrics.csv"

    print("--- Running Model Training as Standalone Script ---")
    
    # Call the function
    metrics = train_and_evaluate_models(
        data_file_path=DEFAULT_DATA_FILE,
    )
    
    if metrics is not None:
        print("\n--- Training Complete ---")
        print("Final Metrics:")
        # print(metrics.to_string())