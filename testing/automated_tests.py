import os
import pandas as pd

# ensures main directories exist
def test_data_paths_exist():
    for d in ["features", "fetch_data", "models", "cleaned_data", 
              "backfilled_data", "training_data"]:
        assert os.path.exists(d), f"Missing directory: {d}"

# checks that cleaned_data/cleaned_backfilled_data.csv (if  it exists) loads correctly."""
def test_feature_csv_structure():
    if os.path.exists("cleaned_data/clean_backfilled_data.csv"):
        df = pd.read_csv("cleaned_data/clean_backfilled_data.csv")
        assert not df.empty, "Dataframe is empty"
        assert "ow_aqi_index" in df.columns, "Missing AQI column"
