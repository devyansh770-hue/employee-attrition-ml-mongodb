"""
data_preprocessing.py
---------------------
Loads the IBM HR Attrition CSV, performs EDA-level checks,
cleans the data, encodes features, and produces train/test splits
suitable for scikit-learn pipelines.

Key design decisions:
- Constant columns (EmployeeCount, Over18, StandardHours) are dropped
  because they carry zero information.
- EmployeeNumber is dropped to prevent data leakage.
- Target variable 'Attrition' is binary-encoded (Yes→1, No→0).
- Categorical features are handled via OneHotEncoder inside the pipeline,
  so we avoid fitting the encoder on the full dataset before splitting.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Path helpers
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_NAME    = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CSV_PATH    = os.path.join(DATASET_DIR, CSV_NAME)

# Columns that are constant / identifiers — removed before ML
COLUMNS_TO_DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
TARGET_COLUMN   = "Attrition"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20


# ─────────────────────────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────────────────────────
def load_dataset(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the IBM HR Attrition CSV into a DataFrame.

    Raises FileNotFoundError with clear instructions if the file is missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at:\n   {path}\n\n"
            "Please download the IBM HR Analytics Employee Attrition dataset from:\n"
            "  https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset\n"
            f"Rename the file to '{CSV_NAME}' and place it in the 'dataset/' folder.\n"
        )
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────
# 2. INSPECTION
# ─────────────────────────────────────────────────────────────────
def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Print and return a summary of the raw dataset quality.

    Returns a dict with keys: shape, dtypes, missing, duplicates, target_dist.
    """
    print("\n" + "="*60)
    print("DATASET INSPECTION REPORT")
    print("="*60)

    shape = df.shape
    print(f"\n📐 Shape          : {shape[0]} rows × {shape[1]} columns")

    # Data types
    dtypes = df.dtypes.to_dict()
    num_numeric = df.select_dtypes(include=np.number).shape[1]
    num_categ   = df.select_dtypes(include="object").shape[1]
    print(f"   Numeric columns : {num_numeric}")
    print(f"   Categorical cols: {num_categ}")

    # Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n🔍 Missing values : {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0])
    else:
        print("   No missing values found.")

    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔁 Duplicate rows : {duplicates}")

    # Target distribution
    target_dist = df[TARGET_COLUMN].value_counts()
    target_pct  = df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(2)
    print(f"\n🎯 Target Distribution ({TARGET_COLUMN}):")
    for label in target_dist.index:
        print(f"   {label}: {target_dist[label]} ({target_pct[label]}%)")

    # Constant columns check
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    print(f"\n⚠️  Constant columns (zero variance): {constant_cols}")

    print("="*60 + "\n")

    return {
        "shape"      : shape,
        "dtypes"     : dtypes,
        "missing"    : missing[missing > 0].to_dict(),
        "duplicates" : duplicates,
        "target_dist": target_dist.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────────────────────────
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows, constant columns, and identifier columns.
    Does NOT encode features (encoding happens inside the ML pipeline).
    """
    df = df.copy()

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"⚠️  Removed {before - after} duplicate row(s).")

    # Drop constant/identifier columns
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"✅ Dropped columns: {cols_to_drop}")

    return df


# ─────────────────────────────────────────────────────────────────
# 4. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────────────────────────
def get_feature_target(df: pd.DataFrame):
    """
    Separate features (X) from the target (y).
    Target is binary-encoded: Yes → 1, No → 0.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series (int, 0/1)
    """
    df = df.copy()
    y  = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    X  = df.drop(columns=[TARGET_COLUMN])
    print(f"✅ Features: {X.shape[1]} columns | Target: {TARGET_COLUMN}")
    return X, y


def get_column_types(X: pd.DataFrame):
    """
    Return lists of numeric and categorical column names.

    Returns
    -------
    numeric_cols   : list[str]
    categorical_cols : list[str]
    """
    numeric_cols     = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    return numeric_cols, categorical_cols


# ─────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Stratified train/test split (80/20).
    Stratification preserves the class imbalance ratio in both sets.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    print(f"✅ Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"   Train attrition rate: {y_train.mean():.2%}")
    print(f"   Test  attrition rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────
def run_preprocessing(path: str = CSV_PATH):
    """
    Orchestrates the full preprocessing workflow.

    Returns
    -------
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, df_clean
    """
    df_raw   = load_dataset(path)
    _        = inspect_dataset(df_raw)
    df_clean = clean_dataset(df_raw)
    X, y     = get_feature_target(df_clean)
    numeric_cols, categorical_cols = get_column_types(X)
    X_train, X_test, y_train, y_test = split_data(X, y)

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols, df_clean


if __name__ == "__main__":
    results = run_preprocessing()
    print("\nPreprocessing complete. Ready for model training.")
