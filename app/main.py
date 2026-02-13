import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

# Load model at startup
model = joblib.load("models/model.pkl")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


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
        "label": label_map[pred]
    }

