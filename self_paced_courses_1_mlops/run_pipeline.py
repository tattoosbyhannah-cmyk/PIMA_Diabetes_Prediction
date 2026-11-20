import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi

# --- 1. DATA PREPARATION ---
DATA_DIR = "self_paced_courses_1_mlops/data"
DATASET_PATH = os.path.join(DATA_DIR, "pima.csv")

print("--- Step 1: Loading and Preparing Data ---")
df = pd.read_csv(DATASET_PATH)

target_col = 'class'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs(DATA_DIR, exist_ok=True)
Xtrain.to_csv(os.path.join(DATA_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(DATA_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(DATA_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(DATA_DIR, "ytest.csv"), index=False)
print(f"Train/test splits successfully written to {DATA_DIR}/")

# --- 2. MODEL TRAINING ---
print("\n--- Step 2: Training Model ---")
MODEL_DIR = "self_paced_courses_1_mlops/deployment/model"

numeric_features = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
preprocessor = make_column_transformer((StandardScaler(), numeric_features))
gb_model = GradientBoostingClassifier(random_state=42)

param_grid = {
    'gradientboostingclassifier__n_estimators': [75, 100, 125],
    'gradientboostingclassifier__max_depth': [2, 3, 4],
    'gradientboostingclassifier__subsample': [0.5, 0.6]
}

model_pipeline = make_pipeline(preprocessor, gb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
print("Best Params Found:\n", grid_search.best_params_)

y_pred_test = best_model.predict(Xtest)
print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "best_pima_diabetes_model_v1.joblib")
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# --- 3. UPLOAD TO HUGGING FACE (Optional Trigger) ---
# This will execute if HF_TOKEN is present in the environment (e.g., inside GitHub Actions)
token = os.environ.get("HF_TOKEN")
if token:
    print("\n--- Step 3: Uploading Deployment Folder to Hugging Face ---")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path="self_paced_courses_1_mlops/deployment",
        repo_id=os.environ.get("HF_SPACE_ID", "hannahmcowan/PIMA_Diabetes_Prediction"),
        repo_type="space",
        delete_patterns=["src/*"],
    )
    print("Deployment folder uploaded successfully to Hugging Face Space!")
else:
    print("\nSkipping Step 3 (HF_TOKEN not found in environment). Run this script inside GitHub Actions to trigger deployment.")
