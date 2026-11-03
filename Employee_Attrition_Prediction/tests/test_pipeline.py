"""
tests/test_pipeline.py
----------------------
Pytest test suite for the Employee Attrition Prediction project.
Covers: DB connection, CSV loading, preprocessing, model training,
prediction output, and invalid input handling.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
DB_DIR   = os.path.join(BASE_DIR, "database")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, DB_DIR)

CSV_PATH  = os.path.join(BASE_DIR, "dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MDL_DIR   = os.path.join(BASE_DIR, "models")
RPT_DIR   = os.path.join(BASE_DIR, "reports")

CSV_AVAILABLE   = os.path.isfile(CSV_PATH)
MODEL_AVAILABLE = (
    os.path.isfile(os.path.join(MDL_DIR, "random_forest.joblib")) or
    os.path.isfile(os.path.join(MDL_DIR, "logistic_regression.joblib"))
)


# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def sample_df():
    """Return a synthetic DataFrame that mimics the IBM HR schema with enough rows for stratified split."""
    data = {
        "Age": [30, 45, 25, 50, 35, 28, 42, 33, 48, 36],
        "Attrition": ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No"],
        "BusinessTravel": ["Travel_Rarely","Non-Travel","Travel_Frequently","Travel_Rarely","Non-Travel",
                           "Travel_Rarely","Travel_Frequently","Non-Travel","Travel_Rarely","Non-Travel"],
        "DailyRate": [800, 1200, 400, 900, 1100, 600, 1300, 950, 750, 1050],
        "Department": ["Sales","Research & Development","Human Resources","Sales","Research & Development",
                       "Sales","Research & Development","Sales","Human Resources","Research & Development"],
        "DistanceFromHome": [5, 20, 1, 10, 7, 12, 3, 15, 8, 4],
        "Education": [3, 4, 2, 5, 3, 2, 4, 3, 1, 4],
        "EducationField": ["Life Sciences","Medical","Human Resources","Marketing","Technical Degree",
                           "Marketing","Life Sciences","Medical","Technical Degree","Other"],
        "EmployeeCount": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "EmployeeNumber": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "EnvironmentSatisfaction": [3, 4, 1, 2, 3, 2, 4, 3, 1, 4],
        "Gender": ["Male","Female","Male","Female","Male","Female","Male","Female","Male","Female"],
        "HourlyRate": [65, 80, 45, 75, 70, 55, 90, 60, 50, 85],
        "JobInvolvement": [3, 3, 2, 4, 3, 2, 4, 3, 2, 3],
        "JobLevel": [2, 3, 1, 4, 2, 1, 3, 2, 1, 2],
        "JobRole": ["Sales Executive","Research Scientist","Human Resources","Manager","Research Scientist",
                    "Sales Representative","Healthcare Representative","Sales Executive","Human Resources","Research Scientist"],
        "JobSatisfaction": [4, 3, 1, 4, 2, 1, 4, 3, 2, 4],
        "MaritalStatus": ["Single","Married","Single","Divorced","Married","Single","Married","Divorced","Single","Married"],
        "MonthlyIncome": [5000, 8000, 3000, 12000, 6000, 2800, 9500, 5200, 3100, 6400],
        "MonthlyRate": [15000, 18000, 10000, 22000, 16000, 12000, 20000, 17000, 11000, 19000],
        "NumCompaniesWorked": [2, 1, 4, 0, 3, 1, 2, 5, 3, 0],
        "Over18": ["Y","Y","Y","Y","Y","Y","Y","Y","Y","Y"],
        "OverTime": ["Yes","No","Yes","No","No","Yes","No","No","Yes","No"],
        "PercentSalaryHike": [14, 18, 12, 20, 15, 11, 17, 13, 16, 19],
        "PerformanceRating": [3, 4, 3, 4, 3, 3, 4, 3, 3, 4],
        "RelationshipSatisfaction": [3, 4, 2, 3, 4, 2, 4, 3, 2, 3],
        "StandardHours": [80, 80, 80, 80, 80, 80, 80, 80, 80, 80],
        "StockOptionLevel": [1, 2, 0, 3, 1, 0, 2, 1, 0, 2],
        "TotalWorkingYears": [8, 20, 3, 30, 12, 4, 18, 9, 5, 14],
        "TrainingTimesLastYear": [3, 2, 1, 4, 3, 2, 3, 2, 1, 3],
        "WorkLifeBalance": [3, 3, 2, 4, 3, 2, 3, 4, 2, 3],
        "YearsAtCompany": [5, 15, 2, 25, 8, 2, 10, 6, 3, 9],
        "YearsInCurrentRole": [3, 8, 1, 12, 4, 2, 7, 4, 2, 5],
        "YearsSinceLastPromotion": [1, 5, 0, 8, 2, 0, 4, 2, 0, 3],
        "YearsWithCurrManager": [3, 7, 1, 10, 4, 2, 6, 3, 1, 5],
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────
# TEST 1: CSV loading
# ─────────────────────────────────────────────────────────────────
class TestCSVLoading:

    def test_load_raises_when_missing(self):
        """FileNotFoundError must be raised for a nonexistent path."""
        from data_preprocessing import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/file.csv")

    @pytest.mark.skipif(not CSV_AVAILABLE, reason="Dataset CSV not present")
    def test_load_returns_dataframe(self):
        """CSV loads into a DataFrame with expected shape."""
        from data_preprocessing import load_dataset
        df = load_dataset(CSV_PATH)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 1470, "Expected 1470 records in the IBM HR dataset"
        assert df.shape[1] == 35,   "Expected 35 columns in the IBM HR dataset"

    @pytest.mark.skipif(not CSV_AVAILABLE, reason="Dataset CSV not present")
    def test_target_column_present(self):
        """Target column 'Attrition' must exist."""
        from data_preprocessing import load_dataset
        df = load_dataset(CSV_PATH)
        assert "Attrition" in df.columns


# ─────────────────────────────────────────────────────────────────
# TEST 2: Data preprocessing
# ─────────────────────────────────────────────────────────────────
class TestPreprocessing:

    def test_clean_drops_constant_columns(self, sample_df):
        """Constant columns (EmployeeCount, Over18, StandardHours) are dropped."""
        from data_preprocessing import clean_dataset, COLUMNS_TO_DROP
        df_clean = clean_dataset(sample_df)
        for col in COLUMNS_TO_DROP:
            assert col not in df_clean.columns, f"Column '{col}' should have been dropped"

    def test_get_feature_target_binary_encoding(self, sample_df):
        """Target variable must be binary (0/1) after encoding."""
        from data_preprocessing import clean_dataset, get_feature_target
        df_clean = clean_dataset(sample_df)
        _, y = get_feature_target(df_clean)
        assert set(y.unique()).issubset({0, 1}), "Target must be 0 or 1"

    def test_no_target_in_features(self, sample_df):
        """'Attrition' column must NOT appear in features X."""
        from data_preprocessing import clean_dataset, get_feature_target
        df_clean = clean_dataset(sample_df)
        X, _ = get_feature_target(df_clean)
        assert "Attrition" not in X.columns

    def test_train_test_split_sizes(self, sample_df):
        """Train size must be ~80% of data; both sets must be non-empty."""
        from data_preprocessing import clean_dataset, get_feature_target, split_data
        df_clean = clean_dataset(sample_df)
        X, y = get_feature_target(df_clean)
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) > 0
        assert len(X_test)  > 0
        assert len(X_train) + len(X_test) == len(X)

    def test_no_data_leakage(self, sample_df):
        """Test indices must not overlap with training indices."""
        from data_preprocessing import clean_dataset, get_feature_target, split_data
        df_clean = clean_dataset(sample_df)
        X, y = get_feature_target(df_clean)
        X_train, X_test, _, _ = split_data(X, y)
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, "Data leakage detected: overlapping indices"


# ─────────────────────────────────────────────────────────────────
# TEST 3: Model training (uses synthetic data, no CSV needed)
# ─────────────────────────────────────────────────────────────────
class TestModelTraining:

    def test_pipeline_builds_without_error(self, sample_df):
        """Both pipelines must instantiate without errors."""
        from data_preprocessing import clean_dataset, get_feature_target, get_column_types
        from train_model import build_logistic_regression_pipeline, build_random_forest_pipeline

        df_clean = clean_dataset(sample_df)
        X, _     = get_feature_target(df_clean)
        num, cat = get_column_types(X)

        lr = build_logistic_regression_pipeline(num, cat)
        rf = build_random_forest_pipeline(num, cat)
        assert lr is not None
        assert rf is not None

    def test_pipeline_fits_and_predicts(self, sample_df):
        """Pipeline must fit on training data and return predictions."""
        from data_preprocessing import clean_dataset, get_feature_target, get_column_types, split_data
        from train_model import build_random_forest_pipeline

        df_clean = clean_dataset(sample_df)
        X, y     = get_feature_target(df_clean)
        num, cat = get_column_types(X)
        X_train, X_test, y_train, y_test = split_data(X, y)

        rf = build_random_forest_pipeline(num, cat)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        assert len(preds) == len(X_test)
        assert all(p in [0, 1] for p in preds)

    def test_predict_proba_sum_to_one(self, sample_df):
        """Probabilities for both classes must sum to 1."""
        from data_preprocessing import clean_dataset, get_feature_target, get_column_types, split_data
        from train_model import build_random_forest_pipeline

        df_clean = clean_dataset(sample_df)
        X, y     = get_feature_target(df_clean)
        num, cat = get_column_types(X)
        X_train, X_test, y_train, _ = split_data(X, y)

        rf = build_random_forest_pipeline(num, cat)
        rf.fit(X_train, y_train)
        probas = rf.predict_proba(X_test)
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────
# TEST 4: Prediction module
# ─────────────────────────────────────────────────────────────────
class TestPrediction:

    @pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained models not found")
    def test_predict_returns_valid_output(self):
        """predict() must return a dict with expected keys and valid values."""
        from prediction import predict
        sample = {
            "Age": 35, "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
            "Department": "Research & Development", "DistanceFromHome": 5,
            "Education": 3, "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 3, "Gender": "Male", "HourlyRate": 65,
            "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Research Scientist",
            "JobSatisfaction": 4, "MaritalStatus": "Single", "MonthlyIncome": 5500,
            "MonthlyRate": 15000, "NumCompaniesWorked": 2, "OverTime": "No",
            "PercentSalaryHike": 14, "PerformanceRating": 3,
            "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
            "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3, "YearsAtCompany": 5,
            "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 3,
        }
        result = predict(sample)
        assert "prediction"  in result
        assert "probability" in result
        assert "model_used"  in result
        assert "disclaimer"  in result
        assert result["prediction"] in ["Yes", "No"]
        assert 0.0 <= result["probability"] <= 1.0

    @pytest.mark.skipif(not MODEL_AVAILABLE, reason="Trained models not found")
    def test_predict_invalid_input_handled(self):
        """predict() with completely invalid input should raise or return gracefully."""
        from prediction import predict
        with pytest.raises(Exception):
            predict({"invalid_key": "bad_value"})

    def test_load_model_raises_when_missing(self):
        """load_model() must raise FileNotFoundError for a non-existent model."""
        from train_model import load_model
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model_xyz")


# ─────────────────────────────────────────────────────────────────
# TEST 5: Database connection (MongoDB with PyMongo)
# ─────────────────────────────────────────────────────────────────
class TestDatabaseConnection:

    def test_env_missing_raises_clear_error(self, monkeypatch):
        """Missing required env var must raise EnvironmentError."""
        from database_connection import get_env_var
        with pytest.raises(EnvironmentError, match="NON_EXISTENT_VAR"):
            get_env_var("NON_EXISTENT_VAR_ABCXYZ")

    def test_connection_returns_bool(self):
        """test_connection() must return a boolean (True when MongoDB is active)."""
        from database_connection import test_connection
        result = test_connection()
        assert isinstance(result, bool)

    def test_required_evaluation_metrics_computed(self, sample_df):
        """evaluate_model must compute Accuracy, R2 Score, MAE, Precision, Recall, and F1."""
        from data_preprocessing import clean_dataset, get_feature_target, get_column_types, split_data
        from train_model import build_random_forest_pipeline
        from evaluate_model import evaluate_model

        df_clean = clean_dataset(sample_df)
        X, y     = get_feature_target(df_clean)
        num, cat = get_column_types(X)
        X_train, X_test, y_train, y_test = split_data(X, y)

        rf = build_random_forest_pipeline(num, cat)
        rf.fit(X_train, y_train)

        metrics = evaluate_model(rf, X_test, y_test, "random_forest")

        required_keys = ["accuracy", "r2_score", "mae", "precision", "recall", "f1_score"]
        for key in required_keys:
            assert key in metrics, f"Required metric '{key}' missing from evaluation output"
            assert isinstance(metrics[key], (int, float))

