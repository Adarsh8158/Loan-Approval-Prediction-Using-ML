import os
from pathlib import Path
import joblib
import pandas as pd

# Path to the trained RandomForest model pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_approval_model.pkl"

# Global cache for the loaded model to avoid reloading on each request
_model_cache = None


def load_model():
    """Load and cache the trained ML pipeline."""
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def calculate_emi(principal: float, tenure_years: float, annual_rate: float = 9.5) -> float:
    """Calculate standard monthly EMI using the standard reducing balance formula."""
    if principal <= 0 or tenure_years <= 0:
        return 0.0
    r = (annual_rate / 12) / 100
    n = tenure_years * 12
    emi = (principal * r * ((1 + r) ** n)) / (((1 + r) ** n) - 1)
    return round(emi, 2)


def predict_loan(data: dict) -> dict:
    """
    Validate input, run prediction via the scikit-learn pipeline,
    and calculate risk metrics and confidence score.
    """
    model = load_model()

    # Normalize categorical string inputs
    education = str(data.get("education", "")).strip().title()
    if education not in ["Graduate", "Not Graduate"]:
        education = "Graduate"

    self_employed = str(data.get("self_employed", "")).strip().capitalize()
    if self_employed not in ["Yes", "No"]:
        self_employed = "No"

    # Type casting with fallback validation
    cibil = int(data.get("cibil_score", 650))
    cibil = max(300, min(900, cibil))

    dependents = int(data.get("no_of_dependents", 0))
    dependents = max(0, min(5, dependents))

    income = max(0.0, float(data.get("income_annum", 0.0)))
    loan_amount = max(0.0, float(data.get("loan_amount", 0.0)))
    loan_term = max(1.0, float(data.get("loan_term", 1.0)))

    res_assets = max(0.0, float(data.get("residential_assets_value", 0.0)))
    com_assets = max(0.0, float(data.get("commercial_assets_value", 0.0)))
    lux_assets = max(0.0, float(data.get("luxury_assets_value", 0.0)))
    bank_assets = max(0.0, float(data.get("bank_asset_value", 0.0)))

    # Total assets
    total_assets = res_assets + com_assets + lux_assets + bank_assets

    # Create input DataFrame matching model feature schema
    input_row = {
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "residential_assets_value": res_assets,
        "commercial_assets_value": com_assets,
        "luxury_assets_value": lux_assets,
        "bank_asset_value": bank_assets,
    }

    df_input = pd.DataFrame([input_row])

    # Run model prediction (1 = Approved, 0 = Rejected)
    raw_pred = int(model.predict(df_input)[0])
    probabilities = model.predict_proba(df_input)[0]

    # Model classes: [0, 1]
    classes = list(model.classes_)
    idx = classes.index(raw_pred)
    confidence = round(float(probabilities[idx]) * 100, 1)
    approval_prob = round(float(probabilities[classes.index(1)]) * 100, 1)

    # Status mapping
    status = "Approved" if raw_pred == 1 else "Rejected"

    # Monthly EMI estimate
    monthly_emi = calculate_emi(loan_amount, loan_term)

    # Risk category calculation
    if cibil >= 750:
        cibil_status = "Excellent"
    elif cibil >= 650:
        cibil_status = "Good"
    elif cibil >= 550:
        cibil_status = "Moderate"
    else:
        cibil_status = "High Risk"

    return {
        "status": status,
        "is_approved": (raw_pred == 1),
        "confidence": confidence,
        "approval_probability": approval_prob,
        "cibil_rating": cibil_status,
        "monthly_emi": monthly_emi,
        "total_assets": total_assets,
        "applicant_summary": {
            "income": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "education": education,
            "self_employed": self_employed
        }
    }
