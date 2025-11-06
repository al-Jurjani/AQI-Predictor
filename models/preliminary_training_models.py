import datetime
import json
import os

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from shap_analysis import generate_shap_analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


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
    print("-" * 21)
    return {"Model": model, "Model Name": name, "RMSE": rmse, "MAE": mae, "R2": r2}


# change threshold to equal the rmse of the best model in the mr in hopsworks
def evaluate_model_performance(metrics, threshold_rmse=50):
    rmse = metrics.get("RMSE", None)
    if rmse is None:
        return False, "RMSE metric missing."

    if rmse <= threshold_rmse:
        return True, f"Model passed: RMSE={rmse:.2f} ≤ {threshold_rmse}"
    else:
        return False, f"Model failed: RMSE={rmse:.2f} > {threshold_rmse}"


# turning this into a single function to be used by automated_training.py
def train_and_evaluate_models(df, test_size=0.2, split_random_state=21):
    # --- Main Function Logic ---
    print("Loading data.")
    df = df

    # Prepare data
    df = df.dropna(axis=1, how="all")
    X = df.drop(
        columns=["ow_aqi_index", "timestamp_utc", "city", "timestamp_key"],
        errors="ignore",
    )
    y = df["ow_aqi_index"]

    X = X.fillna(X.mean())

    TEST_SPLIT = 0.1
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Train size:", X_train.shape, "| Test size:", X_test.shape)
    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=split_random_state
    # )
    # print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    # Define models
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Train and evaluate
    results = []
    best_rmse = np.inf
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        print(f"Evaluating Model: {name}")
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        if result["RMSE"] < best_rmse:
            best_rmse = result["RMSE"]
            # best_mae = result["MAE"]
            # best_r2 = result["R2"]
            best_model_name = result["Model Name"]
            best_model = result["Model"]
        results.append(result)

    # don't see the point for this at the moment
    # passed, eval_msg = evaluate_model_performance(metrics)
    # print(eval_msg)

    # # creating a  timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # # run_dir = f"models/run_{timestamp}"
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
            "MAE": float(
                metrics_df.loc[
                    metrics_df["Model Name"] == best_model_name, "MAE"
                ].values[0]
            ),
            "R2": float(
                metrics_df.loc[
                    metrics_df["Model Name"] == best_model_name, "R2"
                ].values[0]
            ),
        },
    }
    metadata_path = os.path.join(run_dir, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Best model meta data saved at: {metadata_path}")

    # SHAP Analysis
    summary_path, bar_path = generate_shap_analysis(best_model, X_train, run_dir)
    if summary_path and bar_path:
        metadata["shap_summary_plot"] = summary_path
        metadata["shap_bar_plot"] = bar_path
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # # will judge later if really needed, we already have the json file
    # # model_card_path = os.path.join(run_dir, "model_card.txt")
    # # create_model_card(metadata, model_card_path)
    # # best_model = metrics_df.sort_values(by="RMSE").iloc[0]["Model"]

    return metrics_df


# This block allows the script to be run directly
if __name__ == "__main__":
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

    print("--- Running Model Training as Standalone Script ---")

    # Call the function
    metrics = train_and_evaluate_models(
        df,
    )

    if metrics is not None:
        print("\n--- Training Complete ---")
        print("Final Metrics:")
        print(metrics)
