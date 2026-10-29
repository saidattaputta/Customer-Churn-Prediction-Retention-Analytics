import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "saidattaputta/customer-churn-model"
MODEL_FILENAME = "churn_model.pkl"


def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = joblib.load(model_path)

    return model