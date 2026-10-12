
import os
from io import StringIO

import boto3
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

# -------------------------------
# Configuration
# -------------------------------
BUCKET = "mlops-house-prediction"
KEY = "proccessed/2026-09-25/Mlops_house_predication_clean_v1.csv"

MODEL_NAME = "house-price-predictor"
MODEL_ALIAS = "champion"

# If running directly on EC2, use localhost.
# If running inside Docker Compose, use http://mlflow:5000.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("mlops-house-prediction")

s3 = boto3.client("s3")


# -------------------------------
# Fetch processed data from S3
# -------------------------------
def fetch_data():
    response = s3.get_object(Bucket=BUCKET, Key=KEY)
    df = pd.read_csv(
        StringIO(response["Body"].read().decode("utf-8"))
    )
    return df


df = fetch_data()
print(f"Fetched dataset shape: {df.shape}")

# -------------------------------
# Features and target
# -------------------------------
FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score",
]

X = df[FEATURES]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# -------------------------------
# Train and evaluate
# -------------------------------
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

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "data_source": f"s3://{BUCKET}/{KEY}",
    })

    mlflow.log_metrics({
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
    })

    # Log model and register it in the Model Registry.
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Logged model URI: {model_info.model_uri}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 score: {r2:.4f}")

# -------------------------------
# Assign champion alias
# -------------------------------
client = MlflowClient(tracking_uri=TRACKING_URI)

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if not versions:
    raise RuntimeError(
        f"No registered versions found for {MODEL_NAME}"
    )

# Use the version created by this training run.
current_versions = [
    version
    for version in versions
    if version.run_id == run.info.run_id
]

if not current_versions:
    raise RuntimeError(
        "Training succeeded, but the registered model version "
        "could not be found for this run."
    )

latest_version = max(
    current_versions,
    key=lambda version: int(version.version),
)

client.set_registered_model_alias(
    MODEL_NAME,
    MODEL_ALIAS,
    latest_version.version,
)

print(
    f"Registered model: {MODEL_NAME}, "
    f"version: {latest_version.version}, "
    f"alias: {MODEL_ALIAS}"
)

