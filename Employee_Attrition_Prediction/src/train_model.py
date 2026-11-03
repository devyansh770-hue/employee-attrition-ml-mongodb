"""
train_model.py
--------------
Trains Logistic Regression and Random Forest classifiers using
scikit-learn pipelines. Handles class imbalance with class_weight
and saves both models to the models/ folder using Joblib.

Design notes:
- All encoding/scaling is INSIDE the pipeline to prevent data leakage.
- SMOTE is NOT used here to keep sklearn-only dependencies; instead
  class_weight='balanced' is applied within each estimator.
- RandomState is fixed for full reproducibility.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────
# Build sklearn preprocessor
# ─────────────────────────────────────────────────────────────────
def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
      - StandardScales numeric columns
      - OneHotEncodes categorical columns (unknown categories → ignore)
    """
    numeric_transformer     = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,     numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    return preprocessor


# ─────────────────────────────────────────────────────────────────
# Model 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────
def build_logistic_regression_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """Return an untrained Logistic Regression pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(
            max_iter     = 1000,
            class_weight = "balanced",   # handles imbalance
            random_state = RANDOM_STATE,
            solver       = "lbfgs",
        )),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────
# Model 2: Random Forest
# ─────────────────────────────────────────────────────────────────
def build_random_forest_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """Return an untrained Random Forest pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            n_estimators = 200,
            max_depth    = None,
            class_weight = "balanced",   # handles imbalance
            random_state = RANDOM_STATE,
            n_jobs       = -1,
        )),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────
# Train and save
# ─────────────────────────────────────────────────────────────────
def train_and_save(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   numeric_cols: list,
                   categorical_cols: list) -> dict:
    """
    Train both models on the training set and persist them to disk.

    Returns
    -------
    dict with keys 'logistic_regression' and 'random_forest',
    each containing a fitted Pipeline.
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    models = {}

    # ── Logistic Regression ──────────────────────────────────────
    print("\n🔄 Training Logistic Regression …")
    lr_pipe = build_logistic_regression_pipeline(numeric_cols, categorical_cols)
    lr_pipe.fit(X_train, y_train)
    lr_path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
    joblib.dump(lr_pipe, lr_path)
    print(f"   ✅ Saved → {lr_path}")
    models["logistic_regression"] = lr_pipe

    # ── Random Forest ────────────────────────────────────────────
    print("\n🔄 Training Random Forest …")
    rf_pipe = build_random_forest_pipeline(numeric_cols, categorical_cols)
    rf_pipe.fit(X_train, y_train)
    rf_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    joblib.dump(rf_pipe, rf_path)
    print(f"   ✅ Saved → {rf_path}")
    models["random_forest"] = rf_pipe

    print("\n✅ Both models trained and saved.\n")
    return models


# ─────────────────────────────────────────────────────────────────
# Load saved model
# ─────────────────────────────────────────────────────────────────
def load_model(model_name: str) -> Pipeline:
    """
    Load a previously saved model pipeline from disk.

    Parameters
    ----------
    model_name : 'logistic_regression' or 'random_forest'
    """
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run 'python src/train_model.py' first."
        )
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────
# Get feature importance from RF
# ─────────────────────────────────────────────────────────────────
def get_feature_importances(rf_pipeline: Pipeline,
                             numeric_cols: list,
                             categorical_cols: list) -> pd.DataFrame:
    """
    Extract feature importances from the Random Forest pipeline.
    Reconstructs feature names after OneHotEncoding.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_clf        = rf_pipeline.named_steps["classifier"]

    # Get OHE feature names for categorical columns
    ohe          = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
    all_features = numeric_cols + cat_features

    importances = rf_clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature"   : all_features,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import run_preprocessing

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, df_clean = run_preprocessing()
    trained_models = train_and_save(X_train, y_train, numeric_cols, categorical_cols)
    print("Run 'python src/evaluate_model.py' next.")
