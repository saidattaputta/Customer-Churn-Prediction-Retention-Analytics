import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "evaluation_metrics.json"


def main():

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Prepare data using the same process as training
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    df = df.drop(columns=["customerID"])

    if "tenure_group" in df.columns:
        df = df.drop(columns=["tenure_group"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 3. Recreate the same test split
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Load trained model
    model = joblib.load(MODEL_PATH)

    # 5. Generate predictions
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Calculate evaluation metrics
    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision": float(
            precision_score(y_test, y_pred)
        ),
        "recall": float(
            recall_score(y_test, y_pred)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_prob)
        ),
        "confusion_matrix": (
            confusion_matrix(y_test, y_pred).tolist()
        ),
    }

    # 7. Save evaluation results
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    # 8. Print results
    print("\nModel Evaluation Results\n")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print(f"\nMetrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()