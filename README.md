# Customer Churn Prediction & Retention Analytics Platform

An end-to-end machine-learning project for estimating customer churn risk and presenting customer-level predictions and model explanations. The repository combines data preparation, exploratory and statistical analysis, model training, MLflow experiment tracking, SHAP explainability, a FastAPI prediction service, and a Streamlit interface.

## Live Resources

- **Streamlit application:** https://customer-churn-prediction-analytics.streamlit.app/
- **Model artifact:** https://huggingface.co/saidattaputta/customer-churn-model

## Problem

Subscription businesses need to identify customers who may discontinue a service so retention teams can prioritize investigation and outreach. This project uses the IBM Telco Customer Churn dataset to estimate churn probability and expose the factors associated with model predictions.

The output is a decision-support signal. It is not a guarantee that a customer will churn and does not measure the causal impact of a retention offer.

## Dataset

- **Source:** IBM Telco Customer Churn dataset
- **Records:** 7,043
- **Target:** `Churn` (Yes/No)
- **Task:** Binary classification

The features include customer demographics, tenure, contract and service subscriptions, payment method, and charges.

## Workflow

1. Inspect and clean customer records.
2. Explore churn distributions and customer segments.
3. Apply statistical tests to examine selected associations with churn.
4. Encode categorical features and scale numeric variables within a scikit-learn preprocessing pipeline.
5. Train a class-weighted Logistic Regression model using a stratified train/test split.
6. Track parameters, metrics, and model artifacts using MLflow.
7. Generate global and local SHAP explanations.
8. Serve single and batch predictions through FastAPI.
9. Present predictions in a Streamlit application.

## Model Evaluation

Reported test-set metrics from the current README/model workflow:

| Metric | Score |
|---|---:|
| Accuracy | 0.7381 |
| Precision | 0.5043 |
| Recall | 0.7834 |
| F1-score | 0.6136 |
| ROC-AUC | 0.8416 |

The selected operating point prioritizes recall, which identifies more churn-positive cases but also produces false positives. Threshold selection should depend on the cost of unnecessary retention contact versus the cost of missing a customer who leaves.

## Explainability

SHAP is used to examine both overall feature influence and individual predictions. The current analysis highlights tenure, monthly charges, total charges, internet service, and contract type among influential features. These are model associations, not causal drivers.

## API

Run the service locally:

```bash
uvicorn api.main:app --reload
```

The FastAPI service exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Single-customer prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/model-info` | Model metadata |

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Run Locally

```bash
git clone https://github.com/saidattaputta/Customer-Churn-Prediction-Retention-Analytics.git
cd Customer-Churn-Prediction-Retention-Analytics
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

For model training:

```bash
python -m src.training.train
```

For tests:

```bash
PYTHONPATH=. pytest -q
```

The API can be run separately using the command above. Docker configuration is also included in the repository.

## Repository Structure

```text
.
├── api/                 # FastAPI routes and request/response schemas
├── data/                # Raw and processed Telco data
├── models/              # Serialized trained model artifact
├── notebooks/           # Data understanding, EDA, feature engineering
├── src/
│   ├── evaluation/      # Metrics, explainability, quality gate
│   ├── training/        # Model training and MLflow logging
│   ├── batch_predict.py
│   ├── model_loader.py
│   └── predict.py
├── tests/               # API tests
├── app.py               # Streamlit interface
├── Dockerfile
├── requirements.txt
└── README.md
```

## Technology Stack

Python, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, MLflow, SHAP, FastAPI, Pydantic, Streamlit, Pytest, Docker, GitHub Actions, and Hugging Face.

## Limitations

- The model is trained on a single public dataset; performance may not transfer to a different provider or customer population.
- Predicted probabilities depend on the training data and calibration and should not be interpreted as certainty.
- The current project does not implement an always-on retraining and production monitoring pipeline.
- The model does not estimate the causal effect of retention interventions.
- Re-run evaluation and tests when changing the data, preprocessing, model, or deployment artifact.

## Author

Sai Datta Putta · [LinkedIn](https://www.linkedin.com/in/sai-datta-putta/)
