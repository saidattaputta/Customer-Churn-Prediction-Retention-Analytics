# Customer Churn Prediction & Retention Analytics Platform

An end-to-end machine learning platform built to identify customers at risk of churn, understand the factors influencing churn, and support data-driven customer retention decisions.

The project transforms customer data into churn predictions through a complete workflow covering data analysis, statistical testing, machine learning, model explainability, experiment tracking, API development, and deployment.

## Project Results

| Metric | Result |
|---|---:|
| Customers analyzed | 7,043 |
| Model | Logistic Regression |
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1-score | 61.36% |
| ROC-AUC | 84.16% |

### Key Outcomes

- Built an end-to-end churn prediction workflow using customer demographics, subscription information, service usage, and billing data.
- Achieved **78.34% churn recall**, identifying a substantial proportion of customers who actually churned in the test dataset.
- Achieved **84.16% ROC-AUC**, demonstrating the model's ability to distinguish between churn and non-churn customers.
- Applied statistical hypothesis testing to investigate associations between customer characteristics and churn.
- Used SHAP to explain global feature importance and individual customer predictions.
- Developed a Streamlit application and FastAPI service for interactive and programmatic predictions.
- Integrated MLflow experiment tracking and automated API testing.

## Business Problem

Customer churn directly affects subscription-based businesses through lost revenue, reduced customer lifetime value, and increased customer acquisition costs.

Businesses need to identify customers who are likely to leave, understand the factors associated with churn, and prioritize retention efforts.

This project addresses the problem:

> How can historical customer data be used to identify customers at risk of churn and provide interpretable insights to support retention decisions?

The solution predicts customer churn probability and provides analytical and explainability capabilities to help investigate churn risk.

## Solution Overview

The platform combines predictive modeling with customer analytics and model interpretation.

```text
Customer Data
     |
     v
Data Cleaning & Validation
     |
     v
Exploratory Data Analysis
     |
     v
Statistical Hypothesis Testing
     |
     v
Feature Preprocessing
     |
     v
Logistic Regression Model
     |
     +----------------------+
     |                      |
     v                      v
Model Evaluation       MLflow Tracking
     |
     v
SHAP Explainability
     |
     v
Streamlit Application + FastAPI
     |
     v
Customer Churn Predictions
     |
     v
Retention Decision Support
```

## Dataset

The project uses the IBM Telco Customer Churn dataset.

| Dataset Characteristic | Value |
|---|---|
| Customer records | 7,043 |
| Prediction target | `Churn` |
| Target classes | Yes / No |
| Problem type | Binary classification |

The dataset contains customer demographics, account information, subscription details, and service usage.

### Feature Categories

| Category | Example Features |
|---|---|
| Customer demographics | Gender, Senior Citizen, Partner, Dependents |
| Account information | Tenure, Contract, Payment Method |
| Service subscriptions | Internet Service, Online Security, Tech Support |
| Billing | Monthly Charges, Total Charges |

## Analytical Findings

Exploratory data analysis and statistical tests were used to investigate customer churn patterns.

### Statistical Analysis

The project applies:

- Chi-square tests to examine relationships between categorical variables and churn.
- Welch's t-tests to compare numerical customer characteristics across churn groups.

These analyses investigate whether customer attributes exhibit statistically significant associations with churn.

### Customer Churn Drivers

The model's SHAP analysis identifies several features with substantial influence on churn predictions.

| Feature | Mean Absolute SHAP |
|---|---:|
| Tenure | 1.0287 |
| Monthly Charges | 0.6005 |
| Total Charges | 0.4074 |
| Internet Service: Fiber optic | 0.3403 |
| Contract: Month-to-month | 0.3262 |
| Internet Service: DSL | 0.2937 |
| Contract: Two year | 0.2872 |

### Interpretation

The model places substantial importance on:

- Customer tenure.
- Monthly and total charges.
- Internet service type.
- Contract type.

These features help explain how customer characteristics influence predicted churn risk.

**SHAP importance describes model behavior and does not establish that a feature causes churn.**

## Machine Learning Model

### Model Selection

The project uses Logistic Regression as a binary classification model.

The model is implemented within a scikit-learn preprocessing and prediction pipeline.

### Preprocessing

The pipeline includes:

- Numerical feature scaling using `StandardScaler`.
- Categorical feature encoding using `OneHotEncoder`.
- Class weighting to address target class imbalance.

### Training Configuration

| Parameter | Value |
|---|---|
| Train-test split | 80% / 20% |
| Random state | 42 |
| Stratification | Target variable |
| Classifier | Logistic Regression |
| Maximum iterations | 1,000 |

## Model Evaluation

The model was evaluated using a held-out test dataset.

### Performance Metrics

| Metric | Result | Interpretation |
|---|---:|---|
| Accuracy | 73.81% | Overall proportion of correct predictions |
| Precision | 50.43% | Proportion of predicted churners who actually churned |
| Recall | 78.34% | Proportion of actual churners identified |
| F1-score | 61.36% | Balance between precision and recall |
| ROC-AUC | 84.16% | Ability to rank churners above non-churners |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 747 | 288 |
| Actual Churn | 81 | 293 |

### What the Results Mean

The model identifies **293 of the 374 actual churners** in the test dataset.

This corresponds to a churn recall of approximately 78.34%.

The model also flags 288 customers who did not churn, illustrating the trade-off between identifying more potential churners and avoiding unnecessary retention outreach.

The results suggest the model may be useful as a screening tool to prioritize customers for further investigation. Predictions should be combined with customer value, retention campaign costs, and business rules before taking action.

## Classification Threshold Analysis

The platform includes threshold analysis to investigate how prediction decisions change when the classification threshold is adjusted.

Threshold selection affects:

- The number of customers flagged as at risk.
- False-positive and false-negative predictions.
- Potential retention campaign volume.
- The trade-off between identifying churners and limiting unnecessary outreach.

A suitable threshold depends on business objectives and the relative costs of missed churners and unnecessary retention interventions.

## Model Explainability with SHAP

SHAP is used to interpret model predictions and improve transparency.

### Global Explainability

Global SHAP analysis identifies features that contribute most to model predictions across the dataset.

The generated feature importance output provides a ranked view of model influence.

### Local Explainability

Local SHAP analysis explains the contribution of features to an individual customer's prediction.

This helps users investigate questions such as:

- Which features contributed to a high churn probability?
- Which characteristics reduced the predicted churn probability?
- Why did two customers receive different predictions?

### Generated Explainability Artifacts

```text
artifacts/
├── shap_feature_importance.csv
├── shap_summary.png
└── shap_customer_explanation.png
```

## MLflow Experiment Tracking

MLflow is integrated into the training workflow to record model experiments.

Tracked information includes:

- Model parameters.
- Evaluation metrics.
- Trained model artifacts.

The experiment is registered under:

`Customer Churn Prediction`

This provides a foundation for reproducible training runs and model comparison.

## Application and Deployment

The project includes a Streamlit application for interactive customer churn prediction.

The application supports:

- Customer-level prediction.
- Churn probability display.
- Risk classification.
- Model explanation.

The deployed Streamlit application loads its trained model from a Hugging Face model repository.

### Deployment Components

| Component | Purpose |
|---|---|
| Streamlit | Interactive user interface |
| Hugging Face | Model artifact hosting |
| FastAPI | Prediction API |
| Docker | Containerized application |
| GitHub Actions | Automated testing workflow |

## FastAPI Prediction Service

The project includes a FastAPI service for single-customer and batch predictions.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict/batch` | Predict churn for multiple customers |
| GET | `/model-info` | Retrieve model metadata |

### Example Prediction Response

```json
{
  "churn_probability": 0.82,
  "churn_prediction": "Yes"
}
```

The response provides the estimated churn probability and the model's predicted class.

### API Documentation

When running the API locally:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Application | Streamlit |
| API | FastAPI, Uvicorn |
| Validation | Pydantic |
| Testing | Pytest |
| Deployment | Streamlit Community Cloud, Hugging Face |
| Containerization | Docker |
| CI | GitHub Actions |

## Repository Structure

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

## Getting Started

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

## Run the Streamlit Application

```bash
streamlit run app.py
```

Open:

`http://localhost:8501`

## Train the Model

```bash
python -m src.training.train
```

This runs the training workflow, evaluates the model, logs the experiment to MLflow, and saves the trained model.

## Generate SHAP Explanations

```bash
python -m src.evaluation.explain
```

This generates global and local explainability artifacts in the `artifacts/` directory.

## Run the FastAPI Service

```bash
uvicorn api.main:app --reload
```

Open the API documentation:

`http://127.0.0.1:8000/docs`

## Run Tests

```bash
PYTHONPATH=. pytest -q
```

## Run with Docker

Build the Docker image:

```bash
docker build -t customer-churn-app .
```

Run the container:

```bash
docker run -p 8501:8501 customer-churn-app
```

Open:

`http://localhost:8501`

## Business Value

The platform demonstrates how churn predictions and customer analytics can support retention-focused decision-making.

Potential business applications include:

- Prioritizing customers for retention outreach.
- Identifying customer segments with elevated churn risk.
- Investigating churn patterns across contracts and services.
- Supporting targeted retention strategies.
- Providing interpretable evidence to analysts and business stakeholders.

The model does not directly measure the impact of a retention campaign. Its predictions are intended to support decision-making rather than replace business judgment.

## Limitations

- The model is trained on a single public dataset.
- Results may not generalize to other businesses or customer populations.
- Churn probabilities are estimates, not guarantees of customer behavior.
- SHAP explanations describe model behavior and do not establish causal relationships.
- The current implementation does not include a deployed automated retraining or model monitoring pipeline.
- The project does not estimate the causal effect of retention interventions.

## Future Improvements

Potential future extensions include:

- Model monitoring and drift detection.
- Automated retraining pipelines.
- Comparing additional model families.
- Customer lifetime value integration.
- Retention campaign optimization.
- A future experimentation platform to evaluate retention interventions.

## Author

**Sai Datta Putta**
- [LinkedIn](https://www.linkedin.com/in/sai-datta-putta/)
