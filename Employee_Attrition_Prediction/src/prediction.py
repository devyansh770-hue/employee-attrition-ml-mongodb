"""
prediction.py
-------------
Loads the best trained model and provides a predict() function
that accepts a dict of employee attributes and returns:
  - predicted label ('Yes' or 'No')
  - probability of attrition (float, 0-1)

This module is used by both the CLI and the Streamlit dashboard.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import warnings
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")


def get_best_model_name() -> str:
    """
    Read the best model name from the saved evaluation report.
    Falls back to 'random_forest' if the report is missing.
    """
    report_path = os.path.join(REPORT_DIR, "model_evaluation.json")
    if os.path.isfile(report_path):
        with open(report_path) as f:
            report = json.load(f)
        return report.get("best_model", "random_forest")
    return "random_forest"


def load_best_model():
    """Load and return the best model pipeline."""
    name = get_best_model_name()
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Please run: python src/train_model.py"
        )
    model = joblib.load(path)
    print(f"✅ Loaded model: {name}")
    return model, name


def predict(employee_data: dict) -> dict:
    """
    Predict attrition for a single employee.

    Parameters
    ----------
    employee_data : dict
        Keys match the original IBM HR dataset column names (excluding
        constant columns and EmployeeNumber).

    Returns
    -------
    dict with keys:
        'prediction'   : 'Yes' or 'No'
        'probability'  : float between 0 and 1
        'model_used'   : str
        'disclaimer'   : str
    """
    model, model_name = load_best_model()

    # Convert to DataFrame (single row)
    df_input = pd.DataFrame([employee_data])

    # Predict
    label_encoded = model.predict(df_input)[0]        # 0 or 1
    probability   = model.predict_proba(df_input)[0][1]  # P(Attrition=Yes)

    label_str = "Yes" if label_encoded == 1 else "No"

    result = {
        "prediction" : label_str,
        "probability": round(float(probability), 4),
        "model_used" : model_name,
        "disclaimer" : (
            "⚠️ This is a probabilistic model estimate based on historical patterns. "
            "It is NOT a certainty and should be treated as one data point to inform "
            "HR decisions, not as a definitive conclusion about any individual employee."
        ),
    }
    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict attrition for a DataFrame of employees.

    Returns the input DataFrame with two additional columns:
      'Predicted_Attrition'    : 'Yes' or 'No'
      'Attrition_Probability'  : float
    """
    model, _ = load_best_model()
    df = df.copy()
    df["Attrition_Probability"] = model.predict_proba(df)[:, 1].round(4)
    df["Predicted_Attrition"]   = ["Yes" if p >= 0.5 else "No"
                                    for p in df["Attrition_Probability"]]
    return df


# ─────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example employee — values taken from typical IBM HR dataset row
    sample_employee = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5500,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
    }

    result = predict(sample_employee)
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"  Attrition Prediction : {result['prediction']}")
    print(f"  Attrition Probability: {result['probability']:.2%}")
    print(f"  Model Used           : {result['model_used']}")
    print(f"\n{result['disclaimer']}\n")
