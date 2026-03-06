import json
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_ROOT = "models"


def get_latest_model_version():
    versions = []
    for name in os.listdir(MODELS_ROOT):
        path = os.path.join(MODELS_ROOT, name)
        if os.path.isdir(path) and name.startswith("v"):
            try:
                versions.append(int(name[1:]))
            except ValueError:
                pass

    if not versions:
        raise FileNotFoundError("No model versions found in models/")

    latest_version_num = max(versions)
    return f"v{latest_version_num}"


MODEL_VERSION = get_latest_model_version()
MODEL_PATH = f"{MODELS_ROOT}/{MODEL_VERSION}/model.pkl"
METRICS_PATH = f"{MODELS_ROOT}/{MODEL_VERSION}/metrics.json"

app = FastAPI()

# Load latest model at startup
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "model_path": MODEL_PATH,
        "status": "loaded"
    }


@app.get("/metrics")
def get_metrics():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return metrics


@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    pred = int(model.predict(X)[0])

    label_map = {
        0: "malignant",
        1: "benign"
    }

    return {
        "prediction": pred,
        "label": label_map[pred],
        "model_version": MODEL_VERSION
    }
