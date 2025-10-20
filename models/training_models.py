import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data_file_path = "training_data/training_dataset.csv"
output_path = "models/baseline_metrics.csv"

df = pd.read_csv(data_file_path)
X = df.drop(columns=["ow_aqi_index", "timestamp_utc"], errors="ignore")
y = df["ow_aqi_index"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

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

    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}
for name, model in models.items():
    result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results.append(result)

metrics_df = pd.DataFrame(results)
# saving as a csv
metrics_df.to_csv(output_path, index = False)