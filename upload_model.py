from huggingface_hub import HfApi

api = HfApi()

repo_id = "saidattaputta/customer-churn-model"

# Create repository if it does not exist
api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=True,
    exist_ok=True
)

# Upload model
api.upload_file(
    path_or_fileobj="models/churn_model.pkl",
    path_in_repo="churn_model.pkl",
    repo_id=repo_id,
    repo_type="model"
)

print("Model uploaded successfully!")