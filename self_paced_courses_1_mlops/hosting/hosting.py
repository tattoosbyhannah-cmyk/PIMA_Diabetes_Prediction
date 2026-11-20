# Uploads the deployment folder (app, Dockerfile, requirements, model) to the Hugging Face Space
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is not set. Add it as a repository secret under Settings > Secrets and variables > Actions.")

api = HfApi(token=token)
api.upload_folder(
    folder_path="self_paced_courses_1_mlops/deployment",
    repo_id=os.environ.get("HF_SPACE_ID", "hannahmcowan/PIMA_Diabetes_Prediction"),
    repo_type="space",
    # remove the Streamlit template's placeholder app
    delete_patterns=["src/*"],
)
print("Deployment folder uploaded to the Space.")
