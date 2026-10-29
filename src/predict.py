import joblib
import pandas as pd
from pathlib import Path

# Load saved model
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):
    """
    Predict churn probability and churn classification
    for one customer or multiple customers.
    """

    customer_df = pd.DataFrame(customer_data)

    churn_probability = model.predict_proba(customer_df)[:, 1]

    # Default classification threshold
    churn_prediction = (churn_probability >= 0.5).astype(int)

    results = customer_df.copy()
    results["Churn Probability"] = churn_probability
    results["Churn Prediction"] = churn_prediction
    results["Churn Prediction"] = results["Churn Prediction"].map({
        1: "Yes",
        0: "No"
    })

    return results