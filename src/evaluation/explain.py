import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"
DATA_PATH = ROOT_DIR / "data" / "processed" / "telco_churn_clean.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load Model and Dataset
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

# Keep only the features used by the model
feature_names = model.feature_names_in_

X = df[feature_names].copy()


# --------------------------------------------------
# Transform Data
# --------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X)

# Get feature names after preprocessing
transformed_feature_names = preprocessor.get_feature_names_out()


# --------------------------------------------------
# SHAP Explainability
# --------------------------------------------------

explainer = shap.LinearExplainer(
    classifier,
    X_transformed,
)

shap_values = explainer(X_transformed)


# --------------------------------------------------
# Global Feature Importance
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": transformed_feature_names,
    "Mean Absolute SHAP": abs(shap_values.values).mean(axis=0),
})

importance = importance.sort_values(
    by="Mean Absolute SHAP",
    ascending=False,
)

importance.to_csv(
    ARTIFACTS_DIR / "shap_feature_importance.csv",
    index=False,
)


# --------------------------------------------------
# Global SHAP Summary Plot
# --------------------------------------------------

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=transformed_feature_names,
    show=False,
)

plt.tight_layout()

plt.savefig(
    ARTIFACTS_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Individual Customer Explanation
# --------------------------------------------------

customer_index = 0

shap.plots.waterfall(
    shap_values[customer_index],
    max_display=15,
    show=False,
)

plt.tight_layout()

plt.savefig(
    ARTIFACTS_DIR / "shap_customer_explanation.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Output
# --------------------------------------------------

print("SHAP analysis completed successfully.")
print("\nTop 10 important features:")
print(importance.head(10))

print("\nGenerated files:")
print("- artifacts/shap_feature_importance.csv")
print("- artifacts/shap_summary.png")
print("- artifacts/shap_customer_explanation.png")