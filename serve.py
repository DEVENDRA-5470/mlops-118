import os
from pathlib import Path
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

MODEL_URI = "models:/house-price-predictor@champion"

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score",
]


class HouseFeatures(BaseModel):
    sqft: float = Field(..., gt=0, le=20000)
    bedrooms: int = Field(..., gt=0, le=20)
    bathrooms: int = Field(..., gt=0, le=200)
    age_years: int = Field(..., ge=0, le=100)
    garage: int = Field(..., ge=0, le=10)
    location_score: int = Field(..., ge=1, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        app.state.model = mlflow.sklearn.load_model(MODEL_URI)
        print(f"Loaded model: {MODEL_URI}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {MODEL_URI}. "
            f"Check MLflow tracking, model alias and artifacts. "
            f"Details: {exc}"
        ) from exc

    yield


app = FastAPI(
    title="House Price Predictor",
    description="Predict house prices using a registered MLflow model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_URI,
        "tracking_uri": TRACKING_URI,
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    input_df = pd.DataFrame(
        [features.model_dump()],
        columns=FEATURES,
    )

    try:
        prediction = app.state.model.predict(input_df)[0]
        return {"predicted_price": round(float(prediction), 2)}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.get("/")
def frontend():
    index_file = STATIC_DIR / "index.html"

    if not index_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend file static/index.html was not found",
        )

    return FileResponse(index_file)
