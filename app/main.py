import json
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_ROOT = "models"


def get_model_versions():
    versions = []
    for name in os.listdir(MODELS_ROOT):
        path = os.path.join(MODELS_ROOT, name)
        if os.path.isdir(path) and name.startswith("v"):
            try:
                int(name[1:])
                versions.append(name)
            except ValueError:
                pass

    versions.sort(key=lambda x: int(x[1:]))
    return versions


def get_latest_model_version():
    versions = get_model_versions()

    if not versions:
        raise FileNotFoundError("No model versions found in models/")

    return versions[-1]


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


@app.get("/versions")
def list_versions():
    versions = get_model_versions()
    return {
        "available_models": versions,
        "latest": versions[-1] if versions else None
    }


@app.post("/load-model/{version}")
def load_model(version: str):
    global model, MODEL_VERSION, MODEL_PATH, METRICS_PATH

    model_dir = os.path.join(MODELS_ROOT, version)

    if not os.path.exists(model_dir):
        return {"error": "model version not found"}

    model_path = os.path.join(model_dir, "model.pkl")
    metrics_path = os.path.join(model_dir, "metrics.json")

    if not os.path.exists(model_path):
        return {"error": "model.pkl not found for that version"}

    if not os.path.exists(metrics_path):
        return {"error": "metrics.json not found for that version"}

    model = joblib.load(model_path)
    MODEL_VERSION = version
    MODEL_PATH = model_path
    METRICS_PATH = metrics_path

    return {
        "status": "model loaded",
        "active_model": MODEL_VERSION
    }


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
