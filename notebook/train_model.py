"""
Loan Approval Prediction - Model Training & Evaluation Pipeline
This script loads the historical loan dataset, performs preprocessing,
trains a RandomForestClassifier inside a scikit-learn Pipeline, evaluates
metrics, and serializes the trained model for production use.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_approval_dataset.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "loan_approval_model.pkl"


def train():
    print(f"[*] Reading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Clean whitespace from column headers and string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Define features and target
    target_col = "loan_status"
    drop_cols = ["loan_id", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Map target: Approved -> 1, Rejected -> 0
    y = df[target_col].map({"Approved": 1, "Rejected": 0}).astype(int)
    X = df[feature_cols]

    # Split numerical and categorical features
    numerical_cols = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    categorical_cols = ["education", "self_employed"]

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    # Full Model Pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42
        )),
    ])

    # Train/Test Split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[*] Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + "=" * 45)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 45)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"ROC-AUC   : {auc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save trained model
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"[✔] Successfully saved model pipeline to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
