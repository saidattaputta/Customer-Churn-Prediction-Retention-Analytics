from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "Pipeline"
    assert data["features_count"] == 19


def test_single_prediction():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.5,
        "TotalCharges": 666.0,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in ["Yes", "No"]


def test_batch_prediction():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.5,
        "TotalCharges": 666.0,
    }

    response = client.post(
        "/predict/batch",
        json={"customers": [customer, customer]},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert 0 <= prediction["churn_probability"] <= 1
        assert prediction["churn_prediction"] in ["Yes", "No"]


def test_invalid_customer_input():
    response = client.post("/predict", json={"gender": "Female"})

    assert response.status_code == 422