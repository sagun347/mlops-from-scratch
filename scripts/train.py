import os
import json
import joblib
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


MODELS_ROOT = "models"


def get_next_version():
    os.makedirs(MODELS_ROOT, exist_ok=True)

    versions = []
    for name in os.listdir(MODELS_ROOT):
        path = os.path.join(MODELS_ROOT, name)
        if os.path.isdir(path) and name.startswith("v"):
            try:
                versions.append(int(name[1:]))
            except ValueError:
                pass

    next_version_num = 1 if not versions else max(versions) + 1
    return f"v{next_version_num}"


def main():
    model_version = get_next_version()
    model_dir = os.path.join(MODELS_ROOT, model_version)
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Model version: {model_version}")

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "model_version": model_version,
        "training_time": datetime.now().isoformat()
    }

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
