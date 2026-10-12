import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# ----------------------------------
# Configuration
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "Mlops_house_predication_clean_v1.csv"
)

MODEL_NAME = "house-price-predictor"
MODEL_ALIAS = "champion"
EXPERIMENT_NAME = "mlops-house-prediction"

# Training runs directly on EC2 by default.
# For Docker-based training, set MLFLOW_TRACKING_URI=http://mlflow:5000
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score",
]

TARGET = "price"

# ----------------------------------
# MLflow configuration
# ----------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=TRACKING_URI)

# ----------------------------------
# Load local CSV
# ----------------------------------
if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"CSV file not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print(f"Dataset path: {CSV_PATH}")
print(f"Original dataset shape: {df.shape}")

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Convert required columns to numeric values.
df = df[required_columns].copy()

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# Remove invalid numeric values.
df = df.replace(
    [np.inf, -np.inf],
    np.nan,
).dropna()

if len(df) < 5:
    raise ValueError(
        "Not enough valid rows to train the model."
    )

print(f"Valid rows after cleaning: {len(df)}")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ----------------------------------
# Train and evaluate
# ----------------------------------
n_estimators = 150
max_depth = 8

with mlflow.start_run() as run:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )
    )

    r2 = r2_score(y_test, predictions)

    # Log parameters and metrics.
    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "data_source": "local_csv",
        "dataset_rows": len(df),
    })

    mlflow.log_metrics({
        "mae": float(mae),
        "rmse": rmse,
        "r2_score": float(r2),
    })

    # Log the model and register it.
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(2),
    )

    run_id = run.info.run_id

    print("\nTraining completed")
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 score: {r2:.4f}")

# ----------------------------------
# Find the version created by this run
# ----------------------------------
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

current_versions = [
    version
    for version in versions
    if version.run_id == run_id
]

if not current_versions:
    raise RuntimeError(
        f"No registered model version found for run {run_id}."
    )

registered_version = max(
    current_versions,
    key=lambda version: int(version.version),
)

# ----------------------------------
# Assign champion alias
# ----------------------------------
client.set_registered_model_alias(
    MODEL_NAME,
    MODEL_ALIAS,
    registered_version.version,
)

print("\nModel registration successful")
print(f"Model name: {MODEL_NAME}")
print(f"Model version: {registered_version.version}")
print(f"Model alias: {MODEL_ALIAS}")
print("\nTraining pipeline completed successfully.")
