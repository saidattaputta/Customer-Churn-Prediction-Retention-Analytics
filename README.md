# Customer Churn Prediction & Retention Analytics Platform

An end-to-end machine learning platform for predicting customer churn, understanding churn drivers, and supporting data-driven customer retention decisions.

The project combines exploratory data analysis, statistical hypothesis testing, machine learning, experiment tracking, model explainability, API development, and deployment into a practical customer analytics solution.

## Live Demo

- **Streamlit App:** [Customer Churn Prediction App](https://customer-churn-prediction-analytics.streamlit.app/)
- **Model Repository:** [Hugging Face](https://huggingface.co/saidattaputta/customer-churn-model)

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave allows businesses to investigate churn drivers and prioritize retention efforts.

This project develops a predictive solution to:

- Predict customer churn probability.
- Identify customers at higher risk of leaving.
- Analyze factors associated with churn.
- Explain model predictions using SHAP.
- Support individual and batch predictions.
- Translate model predictions into actionable retention insights.

## Business Problem

Businesses often rely on historical customer data to understand churn, but identifying customers who are likely to leave before they do so can be difficult.

The objective is to build a predictive solution that estimates churn risk and helps answer:

- Which customers are most likely to churn?
- What customer characteristics are associated with churn?
- How accurately can churn be predicted?
- Which customers should be prioritized for retention?
- How can predictions support business decisions?

## Dataset

The project uses the **IBM Telco Customer Churn dataset**.

- **Records:** 7,043 customers
- **Target variable:** `Churn`
- **Target classes:** Yes / No
- **Problem type:** Binary classification

The dataset includes customer demographics, account information, subscription details, and service usage.

### Key Features

| Feature Category | Examples |
|---|---|
| Customer demographics | Gender, Senior Citizen, Partner, Dependents |
| Account information | Tenure, Contract, Paperless Billing, Payment Method |
| Service subscriptions | Internet Service, Online Security, Tech Support |
| Charges | Monthly Charges, Total Charges |

## Project Workflow

```text
Business Problem
      |
      v
Data Understanding & Cleaning
      |
      v
Exploratory Data Analysis
      |
      v
Statistical Analysis
      |
      v
Feature Engineering & Preprocessing
      |
      v
Model Training & Evaluation
      |
      v
MLflow Experiment Tracking
      |
      v
Model Explainability with SHAP
      |
      v
Streamlit Application + FastAPI
      |
      v
Deployment
```

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset, customer characteristics, and churn patterns.

Key areas include:

- Overall churn distribution.
- Churn across customer tenure.
- Churn across contract types.
- Churn across payment methods.
- Monthly and total charges.
- Service subscriptions and customer demographics.

## Statistical Analysis

Statistical hypothesis testing was used to examine relationships between customer characteristics and churn.

The analysis includes:

- Chi-square tests for categorical variables.
- Welch's t-tests for numerical variables.
- Statistical interpretation of churn-related differences.

The purpose is to distinguish observed patterns from evidence of statistically significant associations.

## Machine Learning

### Model

The project uses a **Logistic Regression** classifier within a scikit-learn pipeline.

The pipeline includes:

- Numerical feature scaling using `StandardScaler`.
- Categorical feature encoding using `OneHotEncoder`.
- Logistic Regression with class weighting to address class imbalance.

### Training

The dataset is split into training and testing subsets using a stratified split.

| Configuration | Value |
|---|---|
| Test size | 20% |
| Random state | 42 |
| Stratification | Target variable |

### Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7381 |
| Precision | 0.5043 |
| Recall | 0.7834 |
| F1-score | 0.6136 |
| ROC-AUC | 0.8416 |

The model emphasizes churn recall, helping identify a larger proportion of customers who are likely to churn. This comes with a trade-off in precision, meaning some customers predicted to churn may not actually leave.

## Threshold Analysis

The project examines the effect of changing the classification threshold.

A lower threshold can increase the number of customers identified as at risk, while a higher threshold can reduce false positives.

The appropriate threshold depends on business priorities, such as the cost of retention campaigns and the potential value of retaining a customer.

## MLflow Experiment Tracking

MLflow is integrated into the model training workflow to track experiments.

Tracked information includes:

- Model parameters.
- Evaluation metrics.
- Trained model artifacts.

The training script uses the MLflow experiment:

`Customer Churn Prediction`

This supports reproducibility and comparison of model training runs.

## Model Explainability

SHAP is used to interpret the trained Logistic Regression model.

### Global Explainability

Global feature importance helps identify features that have the greatest influence on model predictions across customers.

The top features by mean absolute SHAP value include:

| Feature | Mean Absolute SHAP |
|---|---:|
| Tenure | 1.0287 |
| Monthly Charges | 0.6005 |
| Total Charges | 0.4074 |
| Internet Service: Fiber optic | 0.3403 |
| Contract: Month-to-month | 0.3262 |
| Internet Service: DSL | 0.2937 |
| Contract: Two year | 0.2872 |

### Local Explainability

Local SHAP explanations show how individual customer characteristics contribute to a prediction.

This helps users understand why a specific customer was assigned a higher or lower churn probability.

Generated explainability artifacts:

- `artifacts/shap_feature_importance.csv`
- `artifacts/shap_summary.png`
- `artifacts/shap_customer_explanation.png`

## Application

The Streamlit application allows users to interact with the churn prediction model.

Features include:

- Individual customer churn prediction.
- Churn probability display.
- Risk classification.
- Model explanation.
- Customer-level prediction insights.

The application loads the trained model from Hugging Face.

## API

The FastAPI service exposes endpoints for predictions and model information.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for multiple customers |
| GET | `/model-info` | Retrieve model information |

### Run the API Locally

```bash
uvicorn api.main:app --reload
```

API documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Example Prediction Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5,
  "TotalCharges": 1026.0
}
```

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistics | SciPy |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| App | Streamlit |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Testing | Pytest |
| Deployment | Streamlit Community Cloud, Hugging Face |
| Containerization | Docker |
| CI | GitHub Actions |

## Project Structure

```text
customer-churn-prediction-analytics/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── artifacts/
│   └── # Generated explainability artifacts
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── eda.ipynb
│   └── feature_engineering.ipynb
│
├── src/
│   ├── evaluation/
│   │   └── explain.py
│   ├── training/
│   │   └── train.py
│   ├── batch_predict.py
│   ├── model_loader.py
│   └── predict.py
│
├── tests/
│   └── test_api.py
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saidattaputta/customer-churn-prediction-analytics.git
cd customer-churn-prediction-analytics
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

The app will be available at:

`http://localhost:8501`

## Train the Model

```bash
python -m src.training.train
```

This trains the model, evaluates it, logs the run to MLflow, and saves the model artifact.

## Generate SHAP Explanations

```bash
python -m src.evaluation.explain
```

This generates global and local explainability artifacts in the `artifacts/` directory.

## Run Tests

```bash
PYTHONPATH=. pytest -q
```

## Run with Docker

Build the image:

```bash
docker build -t customer-churn-app .
```

Run the container:

```bash
docker run -p 8501:8501 customer-churn-app
```

Open:

`http://localhost:8501`

## Business Impact

The platform is designed to support retention-focused decisions by:

- Identifying customers with higher churn probabilities.
- Helping prioritize customers for retention outreach.
- Understanding customer attributes associated with churn.
- Supporting data-driven retention strategy development.

The model's predictions should be combined with business context, campaign costs, and customer value before taking retention actions.

## Limitations

- The model is trained on a single public dataset.
- Model performance may vary on data from other businesses.
- Churn probability is a model estimate, not a guarantee of customer behavior.
- The current project does not include a deployed real-time retraining or monitoring pipeline.
- The model does not estimate the causal effect of retention interventions.

## Future Improvements

Potential extensions include:

- Model monitoring and drift detection.
- Automated retraining workflows.
- Testing additional model families.
- Integrating customer lifetime value.
- Retention campaign optimization.
- A future experimentation platform to measure retention intervention effectiveness.

## Author

**Sai Datta Putta**

- [LinkedIn](https://www.linkedin.com/in/sai-datta-putta/)
