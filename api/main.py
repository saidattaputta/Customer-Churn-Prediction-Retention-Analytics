from fastapi import FastAPI, HTTPException
import pandas as pd

from api.schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

from src.predict import predict_churn
from src.batch_predict import predict_batch


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn and generating batch predictions.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerData):
    try:
        customer_data = [customer.model_dump()]
        result = predict_churn(customer_data)

        return PredictionResponse(
            churn_probability=float(result["Churn Probability"].iloc[0]),
            churn_prediction=str(result["Churn Prediction"].iloc[0]),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_customers(request: BatchPredictionRequest):
    try:
        customer_data = pd.DataFrame(
            [customer.model_dump() for customer in request.customers]
        )

        results = predict_batch(customer_data)

        predictions = [
            PredictionResponse(
                churn_probability=float(row["Churn Probability"]),
                churn_prediction=str(row["Churn Prediction"]),
            )
            for _, row in results.iterrows()
        ]

        return BatchPredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/model-info")
def model_info():
    from src.predict import model

    feature_names = (
        model.feature_names_in_.tolist()
        if hasattr(model, "feature_names_in_")
        else []
    )

    return {
        "model_type": type(model).__name__,
        "features_count": len(feature_names),
        "feature_names": feature_names,
        "single_prediction_threshold": 0.5,
        "batch_prediction_threshold": 0.4,
    }