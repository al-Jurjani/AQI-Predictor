# AQI-Predictor

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)

## 📝 Description

Leveraging the power of Python, AQI-Predictor is designed to forecast Air Quality Index (AQI) levels. While no original description was provided, this project aims to deliver a user-friendly tool for predicting air quality, empowering individuals and organizations to make informed decisions regarding their health and activities. Built with Python, the AQI-Predictor will incorporate various features. Stay tuned for more information as the project develops!

## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
requests: latest
pandas: latest
scikit-learn: latest
streamlit: latest
shap: latest
python-dotenv: latest
hopsworks: latest
hopsworks[python]: latest
azure-storage-blob: latest
matplotlib: latest
flake8: latest
autopep8: latest
black: latest
ruff: latest
pre-commit: latest
```

## 📁 Project Structure

```
.
├── AQI_predict-1.pdf
├── app
│   └── frontend.py
├── eda
│   ├── air_quality.ipynb
│   ├── aqi_distribution.png
│   ├── aqi_over_time.png
│   ├── aqi_rolling_avgs.png
│   ├── correlation_heatmap.png
│   ├── df_look.py
│   ├── feature_correlations.csv
│   ├── feature_importance.png
│   ├── pollutants_correlation
│   │   ├── co_vs_aqi.png
│   │   ├── nh3_vs_aqi.png
│   │   ├── no2_vs_aqi.png
│   │   ├── no_vs_aqi.png
│   │   ├── o3_vs_aqi.png
│   │   ├── pm10_vs_aqi.png
│   │   ├── pm2_5_vs_aqi.png
│   │   └── so2_vs_aqi.png
│   ├── shap_analysis.py
│   ├── shap_bar_plot.png
│   └── shap_summary_plot.png
├── features
│   ├── derived_features.py
│   ├── hopsworks_fs.py
│   ├── schema_validator.py
│   ├── tabularize_raw_data.py
│   ├── tabulizer.py
│   └── time_based_features.py
├── fetch_data
│   ├── backfill_data.py
│   ├── fetch_raw_backfill_data.py
│   ├── fetch_raw_data.py
│   ├── karachi_complete_air_quality_data.xlsx
│   ├── karachi_complete_air_quality_data_july_to_dec_2024.csv
│   └── karachi_complete_weather_data.csv
├── models
│   ├── all_features_kept___no_standardization
│   │   ├── baseline_metrics.csv
│   │   └── xgboost_deep_model.pkl
│   ├── all_features_kept___standardization
│   │   └── run_2025-11-08_21-52-03
│   │       ├── baseline_metrics.csv
│   │       └── xgboost_deep_model.pkl
│   ├── automated_training.py
│   ├── model_card_generator.py
│   ├── preliminary_training_models.py
│   ├── removing_low_correlation_features___no_standardization
│   │   ├── baseline_metrics.csv
│   │   └── xgboost_deep_model.pkl
│   ├── removing_low_correlation_features___standardization
│   │   ├── baseline_metrics.csv
│   │   └── xgboost_deep_model.pkl
│   ├── top5_rf_feature_imp___no_standardization
│   │   ├── baseline_metrics.csv
│   │   └── xgboost_deep_model.pkl
│   ├── top5_rf_feature_imp___standardization
│   │   ├── baseline_metrics.csv
│   │   └── xgboost_deep_model.pkl
│   └── training_models.py
├── pyproject.toml
├── raw_data
│   └── archive
│       ├── karachi_weather_data___20251002_155652.json
│       ├── karachi_weather_data___20251002_155656.json
│       ├── karachi_weather_data___20251002_155658.json
│       ├── karachi_weather_data___20251002_155700.json
│       ├── karachi_weather_data___20251002_155702.json
│       ├── karachi_weather_data___20251002_155704.json
│       ├── karachi_weather_data___20251002_155706.json
│       ├── karachi_weather_data___20251004_153431.json
│       ├── karachi_weather_data___20251004_153437.json
│       ├── karachi_weather_data___20251004_153441.json
│       ├── karachi_weather_data___20251004_153445.json
│       ├── karachi_weather_data___20251004_153451.json
│       ├── karachi_weather_data___20251004_153646.json
│       ├── karachi_weather_data___20251004_153650.json
│       ├── karachi_weather_data___20251004_153653.json
│       ├── karachi_weather_data___20251004_153658.json
│       ├── karachi_weather_data___20251004_153800.json
│       ├── karachi_weather_data___20251004_153805.json
│       ├── karachi_weather_data___20251004_153810.json
│       ├── karachi_weather_data___20251004_153815.json
│       ├── karachi_weather_data___20251004_153820.json
│       ├── karachi_weather_data___20251004_154227.json
│       ├── karachi_weather_data___20251004_154231.json
│       ├── karachi_weather_data___20251004_154236.json
│       ├── karachi_weather_data___20251004_154242.json
│       ├── karachi_weather_data___20251004_154248.json
│       ├── karachi_weather_data___20251007_115755.json
│       ├── karachi_weather_data___20251007_115759.json
│       ├── karachi_weather_data___20251007_115834.json
│       ├── karachi_weather_data___20251007_115836.json
│       ├── karachi_weather_data___20251007_115838.json
│       ├── karachi_weather_data___20251015_161703.json
│       ├── karachi_weather_data___20251015_161729.json
│       ├── karachi_weather_data___20251015_161732.json
│       ├── karachi_weather_data___20251015_161736.json
│       ├── karachi_weather_data___20251015_161739.json
│       ├── karachi_weather_data___20251015_161742.json
│       ├── karachi_weather_data___20251020_114118.json
│       ├── karachi_weather_data___20251020_114121.json
│       ├── karachi_weather_data___20251020_114124.json
│       ├── karachi_weather_data___20251020_114126.json
│       ├── karachi_weather_data___20251020_114129.json
│       ├── karachi_weather_data___20251020_114132.json
│       ├── karachi_weather_data___20251020_114802.json
│       ├── karachi_weather_data___20251020_114805.json
│       ├── karachi_weather_data___20251020_114808.json
│       ├── karachi_weather_data___20251020_114810.json
│       ├── karachi_weather_data___20251022_112001.json
│       ├── karachi_weather_data___20251022_112006.json
│       ├── karachi_weather_data___20251022_112012.json
│       ├── karachi_weather_data___20251022_112157.json
│       ├── karachi_weather_data___20251022_112201.json
│       ├── karachi_weather_data___20251022_112204.json
│       ├── karachi_weather_data___20251022_112738.json
│       ├── karachi_weather_data___20251022_112741.json
│       ├── karachi_weather_data___20251022_112744.json
│       ├── karachi_weather_data___20251023_053628.json
│       ├── karachi_weather_data___20251023_054058.json
│       ├── karachi_weather_data___20251023_054331.json
│       ├── karachi_weather_data___20251023_055903.json
│       ├── karachi_weather_data___20251023_062120.json
│       ├── karachi_weather_data___20251023_071430.json
│       ├── karachi_weather_data___20251023_081945.json
│       ├── karachi_weather_data___20251023_091540.json
│       ├── karachi_weather_data___20251023_101551.json
│       ├── karachi_weather_data___20251023_111245.json
│       ├── karachi_weather_data___20251023_122844.json
│       ├── karachi_weather_data___20251023_132802.json
│       ├── karachi_weather_data___20251023_141455.json
│       ├── karachi_weather_data___20251023_151457.json
│       ├── karachi_weather_data___20251023_161728.json
│       ├── karachi_weather_data___20251023_171250.json
│       ├── karachi_weather_data___20251023_182009.json
│       ├── karachi_weather_data___20251023_191127.json
│       ├── karachi_weather_data___20251023_201444.json
│       ├── karachi_weather_data___20251023_211226.json
│       ├── karachi_weather_data___20251023_221310.json
│       ├── karachi_weather_data___20251023_231240.json
│       ├── karachi_weather_data___20251024_005910.json
│       ├── karachi_weather_data___20251024_022320.json
│       ├── karachi_weather_data___20251024_033049.json
│       ├── karachi_weather_data___20251024_041639.json
│       ├── karachi_weather_data___20251024_051424.json
│       ├── karachi_weather_data___20251024_062015.json
│       ├── karachi_weather_data___20251024_071355.json
│       ├── karachi_weather_data___20251024_081855.json
│       ├── karachi_weather_data___20251024_091609.json
│       ├── karachi_weather_data___20251024_101543.json
│       ├── karachi_weather_data___20251024_111244.json
│       ├── karachi_weather_data___20251024_122847.json
│       ├── karachi_weather_data___20251024_132707.json
│       ├── karachi_weather_data___20251024_141450.json
│       ├── karachi_weather_data___20251024_151445.json
│       ├── karachi_weather_data___20251024_161719.json
│       ├── karachi_weather_data___20251024_171248.json
│       ├── karachi_weather_data___20251024_182024.json
│       ├── karachi_weather_data___20251024_191122.json
│       ├── karachi_weather_data___20251024_201548.json
│       ├── karachi_weather_data___20251024_211222.json
│       ├── karachi_weather_data___20251024_221339.json
│       ├── karachi_weather_data___20251024_231312.json
│       ├── karachi_weather_data___20251025_010044.json
│       ├── karachi_weather_data___20251025_022658.json
│       ├── karachi_weather_data___20251025_033211.json
│       ├── karachi_weather_data___20251025_041602.json
│       ├── karachi_weather_data___20251025_051326.json
│       ├── karachi_weather_data___20251025_061833.json
│       ├── karachi_weather_data___20251025_071250.json
│       ├── karachi_weather_data___20251025_081608.json
│       ├── karachi_weather_data___20251025_091303.json
│       ├── karachi_weather_data___20251025_101303.json
│       ├── karachi_weather_data___20251025_111041.json
│       ├── karachi_weather_data___20251025_122342.json
│       ├── karachi_weather_data___20251025_131904.json
│       ├── karachi_weather_data___20251025_141144.json
│       ├── karachi_weather_data___20251025_151214.json
│       ├── karachi_weather_data___20251025_161512.json
│       ├── karachi_weather_data___20251025_171110.json
│       ├── karachi_weather_data___20251025_181740.json
│       ├── karachi_weather_data___20251025_191028.json
│       ├── karachi_weather_data___20251025_201430.json
│       ├── karachi_weather_data___20251025_211148.json
│       ├── karachi_weather_data___20251025_221202.json
│       ├── karachi_weather_data___20251025_231201.json
│       ├── karachi_weather_data___20251026_010650.json
│       ├── karachi_weather_data___20251026_030609.json
│       ├── karachi_weather_data___20251026_041612.json
│       ├── karachi_weather_data___20251026_051333.json
│       ├── karachi_weather_data___20251026_061911.json
│       └── karachi_weather_data___20251026_071240.json
├── requirements.txt
└── testing
    ├── automated_tests.py
    ├── test_OW_api.py
    └── test_hop_api.py
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/al-Jurjani/AQI-Predictor.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request

Please ensure your code follows the project's style guidelines and includes tests where applicable.

---
*This README was generated with ❤️ by ReadmeBuddy*
