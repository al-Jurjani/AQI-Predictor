import datetime
import json
import os

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
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


# turning this into a single function to be used by automated_training.py
def train_and_evaluate_models(df, test_size=0.2, split_random_state=21):
    # --- Main Function Logic ---
    print("Loading data.")
    df = df
    print("DF shape before preprocessing: ", df.shape)

    # Prepare data
    df = df.dropna(axis=1, how="all")
    print("DF shape after preprocessing: ", df.shape)
    # X = df.drop(
    #     # columns=["ow_aqi_index", "timestamp_utc", "city", "timestamp_key"],
    #     columns = [
    #         "ow_aqi_index",
    #         "timestamp_utc",
    #         "city",
    #         "timestamp_key",
    #         "temp_feels_like",
    #         "co_no2_ratio",
    #         "temp_rolling_avg_4h",
    #         "temp_rolling_avg_30d",
    #         "temp_rolling_avg_7d",
    #         "temp_rolling_avg_24h",
    #         "so2_no2_ratio",
    #         "temp",
    #         "so2_wind_disp",
    #         "co_wind_disp",
    #         "nh3_wind_disp",
    #         "pm2_5_wind_disp",
    #         "no2_wind_disp",
    #         "no_wind_disp",
    #         "pm10_wind_disp",
    #         "o3_wind_disp"
    #     ],
    #     errors="ignore",
    # )
    top_5_features = [
        "pm10",
        "pm2_5",
        "temp_rolling_avg_30d",
        "aqi_pct_change_24h",
        "o3",
    ]
    X = df[top_5_features]
    y = df["ow_aqi_index"]

    # X = X.fillna(X.mean())

    TEST_SPLIT = 0.2
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    # # 2. Create and Fit Scaler ON TRAINING DATA ONLY
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    # # 3. Transform Test Data (using the scaler fitted on train)
    # X_test = scaler.transform(X_test)

    print("Train size:", X_train.shape, "| Test size:", X_test.shape)

    models = {
        "DecisionTree_shallow": DecisionTreeRegressor(max_depth=3, random_state=1),
        "DecisionTree_deep": DecisionTreeRegressor(max_depth=8, random_state=2),
        "LinearRegression": LinearRegression(n_jobs=-1),
        "Ridge_alpha_1": Ridge(alpha=1.0),
        "Ridge_low_alpha": Ridge(alpha=0.1, random_state=5),
        "Ridge_high_alpha": Ridge(alpha=10.0, random_state=6),
        "Lasso_alpha_1": Lasso(alpha=1.0),
        "Lasso_low_alpha": Lasso(alpha=0.1, random_state=8),
        "Lasso_high_alpha": Lasso(alpha=1.0, random_state=9),
        "KNeighbors_uniform": KNeighborsRegressor(
            n_neighbors=5, weights="uniform", n_jobs=-1
        ),
        "KNeighbors_distance": KNeighborsRegressor(
            n_neighbors=10, weights="distance", n_jobs=-1
        ),
        # 'SVR_linear': SVR(kernel='linear', C=1.0),
        # 'SVR_rbf': SVR(kernel='rbf', C=10.0, gamma='scale'), # 'scale' is a good default
        "RandomForest_deep": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=14, n_jobs=-1
        ),
        "RandomForest_shallow": RandomForestRegressor(
            n_estimators=150, max_depth=3, random_state=15, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100, max_depth=10, random_state=16, n_jobs=-1
        ),
        "XGBoost_deep": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.01,
            random_state=17,
            n_jobs=-1,
        ),
        "XGBoost_shallow": XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            random_state=18,
            n_jobs=-1,
        ),
        "LightGBM_default": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=-1,
            random_state=19,
            n_jobs=-1,
        ),
        "LightGBM_faster": LGBMRegressor(
            n_estimators=100, learning_rate=0.03, random_state=20, n_jobs=-1
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
            best_mae = result["MAE"]
            best_r2 = result["R2"]
            best_model_name = result["Model Name"]
            best_model = result["Model"]
        results.append(result)

    # # creating a  timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = "models/top5_rf_feature_imp___no_standardization"
    os.makedirs(run_dir, exist_ok=True)

    metrics_df = pd.DataFrame(results)
    metrics_df.sort_values(by="RMSE", ascending=True, inplace=True)
    metrics_csv_path = f"{run_dir}/baseline_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Training model metrics saved at: {metrics_csv_path}")

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
            "MAE": float(best_mae),
            "R2": float(best_r2),
        },
    }
    metadata_path = os.path.join(run_dir, "best_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Best model meta data saved at: {metadata_path}")

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
        print(metrics.to_string())
