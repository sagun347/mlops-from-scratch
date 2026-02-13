import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer


def main():
    os.makedirs("models", exist_ok=True)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print("Label meanings:") 
    print("0 =", data.target_names[0])
    print("1 =", data.target_names[1])
	
    joblib.dump(model, "models/model.pkl")
    print("Saved model to models/model.pkl")


if __name__ == "__main__":
    main()

