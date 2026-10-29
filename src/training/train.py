import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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


def main():

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Data preparation
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove identifier and EDA-only feature
    df = df.drop(columns=["customerID"])

    if "tenure_group" in df.columns:
        df = df.drop(columns=["tenure_group"])

    # 3. Define features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # 6. Selected model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    # 7. Train
    model.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # 9. MLflow tracking
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run(run_name="Automated_Weighted_Logistic_Regression"):

        mlflow.log_param(
            "model",
            "Logistic Regression"
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
        )

        mlflow.log_param(
            "max_iter",
            1000
        )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="churn_model"
        )

    # 10. Save model artifact
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    # 11. Display results
    print("\nModel training completed!")
    print("\nEvaluation metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()