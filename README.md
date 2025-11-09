# AQI-Predictor

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)
![10Pearls](https://10pearls.com/)

# A Data Science Project

This project is a complete solution for predicting the Air Quality Index (AQI) for the city of Karachi, using weather and air quality data from the OpenWeather API. It leverages a modern stack to automate data fetching, model training, and deployment, and includes a user-friendly frontend made by streamlit to visualize the predictions.

The project was done during an internship in 10Pearls' Shine Program.

## 🚀 Features

- **Automated Data Fetching**: A GitHub Actions workflow runs on a schedule to fetch hourly weather and air quality data from the OpenWeather API.
- **Cloud Storage**: Fetched data is stored as JSON files in Azure Blob Storage, ensuring a scalable and reliable data source.
- **Exploratory Data Analysis (EDA)**: A comprehensive analysis of the data is performed to understand the relationships between different pollutants and the AQI. This includes visualizations of data distributions, time-series trends, and feature correlations.
- **Model Training and Evaluation**: Daily automated model training from a selection of a set few models - best model is used for predictions for that day!
- **CI/CD with GitHub Actions**: The project utilizes three automated workflows:
    1.  A CI pipeline for running linters (`black`, `ruff`) and tests.
    2.  A scheduled workflow for fetching data.
    3.  A workflow for automated model training and evaluation.
- **Pre-commit Hooks**: The project uses pre-commit hooks with `black`, `ruff`, and other linters to maintain code quality and consistency.
- **Interactive Frontend**: A web application built with Streamlit provides a user-friendly interface to get AQI predictions and view data visualizations.
- **Cloud Deployment**: The Streamlit application is designed to be hosted on an Azure Virtual Machine, making it accessible to a wider audience (Note: This is a planned feature).
```

## 📁 Repository Structure

The repository is organized to ensure a clear separation of concerns, making the project modular and maintainable.

```
├── .github/workflows/ # GitHub Actions workflows for CI, data fetching, and model training
├── app/ # Source code for the Streamlit frontend
│ └── frontend.py
├── eda/ # Jupyter notebooks and plots for Exploratory Data Analysis
│ ├── aqi_distribution.png
│ ├── correlation_heatmap.png
│ └── ...
├── features/ # Scripts for feature engineering
│ ├── derived_features.py
│ └── time_based_features.py
├── fetch_data/ # Scripts for fetching data from APIs and storing it
│ └── fetch_raw_data.py
├── models/ # Scripts for training models, saved models, and performance metrics
│ ├── training_models.py
│ ├── automated_training.py
│ └── xgboost_deep_model.pkl
├── raw_data/ # Local storage for raw data (usually gitignored)
├── testing/ # Automated tests for the codebase
│ └── test_OW_api.py
├── .gitignore
├── .pre-commit-config.yaml # Configuration for pre-commit hooks
├── requirements.txt # Project dependencies
└── README.md
```

## 🛠️ Tech Stack

- **Python**: The primary programming language.
- **Scikit-learn** & **XGBoost**: For machine learning model development.
- **Pandas** & **NumPy**: For data manipulation and numerical operations.
- **Streamlit**: For building the interactive web frontend.
- **GitHub Actions**: For CI/CD and automation.
- **Azure Blob Storage**: For cloud data storage.
- **Azure VMs**: For cloud deployment of the application.
- **OpenWeather API**: As the source of weather and air quality data.
- **Black** & **Ruff**: For code formatting and linting.

## ⚙️ Getting Started

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/al-Jurjani/AQI-Predictor.git
cd AQI-Predictor
```

### 2. Set Up a Virtual Environment
It's recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a .env file in the root directory and add the following, and include your secrets in it!

### 5. Run the Streamlit Application
To launch the frontend, run the following command:

```bash
streamlit run app/frontend.py
```

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
