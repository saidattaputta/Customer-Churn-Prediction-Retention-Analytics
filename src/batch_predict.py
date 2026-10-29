
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Batch Prediction Function
# --------------------------------------------------

def predict_batch(customer_data, threshold=0.4):
    """
    Generate churn predictions for multiple customers.

    Parameters
    ----------
    customer_data : pandas.DataFrame
        Customer data containing the model's input features.

    threshold : float
        Classification threshold for churn prediction.

    Returns
    -------
    pandas.DataFrame
        Original customer data with churn probabilities
        and churn predictions.
    """

    # Validate input
    if customer_data.empty:
        raise ValueError("The uploaded customer dataset is empty.")

    # Predict churn probabilities
    churn_probabilities = model.predict_proba(
        customer_data
    )[:, 1]

    # Predict churn labels using the chosen threshold
    churn_predictions = (
        churn_probabilities >= threshold
    ).astype(int)

    # Copy input data to avoid modifying original
    results = customer_data.copy()

    # Add prediction results
    results["Churn Probability"] = churn_probabilities

    results["Churn Prediction"] = (
        pd.Series(churn_predictions)
        .map({
            1: "Yes",
            0: "No"
        })
        .values
    )

    return results
