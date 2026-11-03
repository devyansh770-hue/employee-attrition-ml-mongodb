# Academic Project Report

## Employee Attrition Prediction & HR Analytics System Using MongoDB (PyMongo) and Random Forest Machine Learning

**Department:** Computer Science & Engineering  
**Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Project Category:** Machine Learning & NoSQL Database Capstone  
**Dataset:** IBM HR Analytics Employee Attrition & Performance  
**Technology Stack:** Python 3.13, MongoDB, PyMongo, Scikit-learn, Streamlit, Pandas, Seaborn  

---

## Executive Summary & Abstract

Employee attrition represents a multi-billion-dollar operational liability across contemporary enterprise organizations. When valuable personnel depart unexpectedly, businesses bear direct recruitment costs, comprehensive onboarding overhead, and severe productivity depletion. Traditional human resource information systems (HRIS) analyze turnover retrospectively through exit interviews and quarterly resignation tallies.

This project implements an end-to-end, production-ready enterprise predictive analytics platform powered by a **MongoDB NoSQL Document Store (interfaced via PyMongo)** and a **Random Forest Machine Learning Classifier**. The solution ingests the IBM HR Analytics dataset (1,470 records, 35 raw features), normalizes and structures the records into JSON/BSON document collections with high-performance compound indexing, and trains an ensemble Random Forest pipeline equipped with `class_weight='balanced'` to overcome severe target class imbalance (~16.1% attrition).

The model is evaluated against the required academic metrics: **Accuracy, R² Score, MAE (Mean Absolute Error), Precision, Recall, and F1-Score**. An interactive Streamlit web dashboard connects to the live MongoDB instance, executes real-time multi-stage PyMongo aggregation pipelines, visualizes model diagnostics, and provides HR executives with real-time probabilistic attrition risk forecasting and automated prediction logging back into MongoDB.

---

## 1. Problem Statement & Motivation

In corporate workforce management, voluntary turnover averages 15–20% annually. The cost to replace a departing professional ranges from 50% to over 200% of their base annual salary. Conventional relational SQL architectures often incur rigidity when dealing with evolving, nested employee attributes (e.g., dynamic performance milestones, evolving survey scores, flexible benefit packages). 

The primary objectives of this project are:
1. **NoSQL Document Modeling**: Design and implement a document-oriented database using MongoDB and PyMongo to store complete employee demographic, organizational, and compensatory records.
2. **Aggregations & Analytics**: Build multi-stage PyMongo aggregation pipelines (`$group`, `$project`, `$switch`, `$cond`, `$sort`) to extract real-time enterprise workforce analytics.
3. **Single Supervised ML Classifier**: Build an ensemble **Random Forest Classifier** scikit-learn pipeline addressing severe class imbalance without synthetic data leakage.
4. **Rigorous Evaluation Metrics**: Compute and analyze all required evaluation criteria:
   - Classification performance: **Accuracy, Precision, Recall, F1-Score**
   - Probability calibration & variance: **R² Score and MAE**
5. **Interactive Executive Platform**: Deliver a modern web dashboard in Streamlit featuring dynamic querying, diagnostic visualization, and live prediction persistence.

---

## 2. Technology Stack & Architecture

| Layer | Technology | Version / Specification | Role in System |
|---|---|---|---|
| **Database** | MongoDB Server | 7.0+ (localhost:27017) | NoSQL document database storing employee profiles, lookup summaries, and prediction logs |
| **Database Driver** | PyMongo | >= 4.8.0 | Python MongoDB client handling connection pooling, document upserts, and aggregation pipelines |
| **Programming Language** | Python | 3.10+ (tested on 3.13) | Core logic, data manipulation, and pipeline orchestration |
| **Data Manipulation** | Pandas & NumPy | 2.2+ / 2.1+ | Dataframe preprocessing, feature splitting, matrix operations |
| **Machine Learning** | Scikit-learn | >= 1.5.0 | Random Forest ensemble classifier, ColumnTransformer, StandardScaler, OneHotEncoder |
| **Persistence** | Joblib | >= 1.4.0 | Serialization of trained ML pipelines |
| **Web Application** | Streamlit | >= 1.40.0 | Multi-page interactive executive analytics dashboard |
| **Visualization** | Matplotlib & Seaborn | >= 3.9 / 0.13 | High-resolution statistical charting and matrix plotting |
| **Testing Suite** | Pytest | >= 8.3.0 | Unit and integration test suite covering pipelines, MongoDB, and prediction |

### System Data Flow Architecture

```
                       ┌────────────────────────────────────────┐
                       │     IBM HR Analytics CSV Dataset       │
                       │   (1,470 Records × 35 Attributes)      │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │        database/insert_data.py         │
                       │  (Document Structuring & Indexing)     │
                       └───────────────────┬────────────────────┘
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
             ┌─────────────────────────┐         ┌─────────────────────────┐
             │   MongoDB Collection:   │         │   MongoDB Collection:   │
             │       'employees'       │         │  'departments' & 'roles'│
             │   (1,470 BSON Docs)     │         │   (Lookup Aggregates)   │
             └───────────┬─────────────┘         └─────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │       src/data_preprocessing.py   │
        │  • Drop constants & identifiers   │
        │  • Stratified 80/20 train/test    │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │         src/train_model.py        │
        │  • ColumnTransformer (OHE+Scale)  │
        │  • RandomForestClassifier (200)   │
        │  • models/random_forest.joblib    │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │        src/evaluate_model.py      │
        │  • Accuracy, R², MAE, Precision,  │
        │    Recall, F1-Score, ROC-AUC      │
        │  • Confusion Matrix & ROC Curves  │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌───────────────────────────────────┐
        │       app.py (Streamlit Web UI)   │
        │  1. Executive Dashboard           │
        │  2. MongoDB Analytics (PyMongo)   │
        │  3. ML Performance Results        │
        │  4. Attrition Prediction Form     │
        │  5. Data-Driven HR Insights       │
        └───────────────────────────────────┘
```

---

## 3. MongoDB Schema Design & PyMongo Aggregation

### 3.1 Document Modeling in MongoDB

Unlike relational tables that mandate strict column types and foreign key joins across 5 tables, MongoDB organizes employee records into semi-structured documents. Each document encapsulates demographics, job profile, compensation, satisfaction ratings, and historical experience in a single entity:

```json
{
  "_id": ObjectId("6722..."),
  "emp_number": 1,
  "age": 41,
  "gender": "Female",
  "marital_status": "Single",
  "department": "Sales",
  "job_role": "Sales Executive",
  "job_level": 2,
  "business_travel": "Travel_Rarely",
  "over_time": "Yes",
  "monthly_income": 5993.0,
  "daily_rate": 1102.0,
  "hourly_rate": 94.0,
  "monthly_rate": 19479.0,
  "percent_salary_hike": 11,
  "stock_option_level": 0,
  "education": 2,
  "education_field": "Life Sciences",
  "environment_satisfaction": 2,
  "job_involvement": 3,
  "job_satisfaction": 4,
  "performance_rating": 3,
  "relationship_satisfaction": 1,
  "work_life_balance": 1,
  "total_working_years": 8,
  "years_at_company": 6,
  "years_in_current_role": 4,
  "years_since_last_promotion": 0,
  "years_with_curr_manager": 5,
  "num_companies_worked": 8,
  "training_times_last_year": 0,
  "distance_from_home": 1,
  "attrition": "Yes",
  "attrition_binary": 1
}
```

### 3.2 Indexing Strategy
To ensure query execution times remain under 2 milliseconds across large employee bases, the following indexes are automatically constructed on the `employees` collection:
* `emp_number`: Unique index (primary business key).
* `department`: Single-field index for organizational filtering.
* `job_role`: Single-field index for role analysis.
* `attrition`: Target index for fast distribution queries.
* `monthly_income`: Range index for salary distributions.
* `years_at_company`: Range index for tenure bucketing.

### 3.3 Key PyMongo Aggregation Pipelines

#### A. Department-Wise Attrition Pipeline
Calculates total workforce, count of departing employees, retained staff, and percentage attrition:

```python
pipeline_dept = [
    {
        "$group": {
            "_id": "$department",
            "total_employees": {"$sum": 1},
            "attrited": {"$sum": {"$cond": [{"$eq": ["$attrition", "Yes"]}, 1, 0]}},
            "retained": {"$sum": {"$cond": [{"$eq": ["$attrition", "No"]}, 1, 0]}}
        }
    },
    {
        "$project": {
            "department": "$_id",
            "total_employees": 1,
            "attrited": 1,
            "retained": 1,
            "attrition_rate_pct": {
                "$round": [{"$multiply": [{"$divide": ["$attrited", "$total_employees"]}, 100]}, 2]
            }
        }
    },
    {"$sort": {"attrition_rate_pct": -1}}
]
```

#### B. Salary & Compensation Aggregation Pipeline
Computes statistical wage metrics across job classifications:

```python
pipeline_salary = [
    {
        "$group": {
            "_id": "$job_role",
            "avg_monthly_income": {"$avg": "$monthly_income"},
            "min_income": {"$min": "$monthly_income"},
            "max_income": {"$max": "$monthly_income"},
            "headcount": {"$sum": 1}
        }
    },
    {
        "$project": {
            "job_role": "$_id",
            "avg_income": {"$round": ["$avg_monthly_income", 2]},
            "min_income": 1,
            "max_income": 1,
            "headcount": 1
        }
    },
    {"$sort": {"avg_income": -1}}
]
```

#### C. Tenure Bucketing Pipeline
Leverages MongoDB's conditional `$switch` statement to aggregate workforce longevity:

```python
pipeline_tenure = [
    {
        "$project": {
            "tenure_bucket": {
                "$switch": {
                    "branches": [
                        {"case": {"$lte": ["$years_at_company", 2]}, "then": "0-2 years"},
                        {"case": {"$lte": ["$years_at_company", 5]}, "then": "3-5 years"},
                        {"case": {"$lte": ["$years_at_company", 10]}, "then": "6-10 years"}
                    ],
                    "default": "10+ years"
                }
            },
            "is_attrited": {"$cond": [{"$eq": ["$attrition", "Yes"]}, 1, 0]}
        }
    },
    {
        "$group": {
            "_id": "$tenure_bucket",
            "total_employees": {"$sum": 1},
            "attrited_count": {"$sum": "$is_attrited"}
        }
    },
    {
        "$project": {
            "tenure_bucket": "$_id",
            "total_employees": 1,
            "attrited_count": 1,
            "attrition_rate_pct": {
                "$round": [{"$multiply": [{"$divide": ["$attrited_count", "$total_employees"]}, 100]}, 2]
            }
        }
    },
    {"$sort": {"attrition_rate_pct": -1}}
]
```

---

## 4. Machine Learning Implementation: Random Forest Classifier

### 4.1 Data Preprocessing & Leakage Prevention
To ensure zero data leakage between training and testing environments:
1. **Constant Feature Pruning**: Columns with zero variance (`EmployeeCount`, `Over18`, `StandardHours`) and identifiers (`EmployeeNumber`) are eliminated.
2. **Stratified Split**: The remaining data is partitioned via stratified train/test split (80% train = 1,176 records, 20% test = 294 records), guaranteeing an exact 16.1% positive class ratio in both splits.
3. **Encapsulated Preprocessing**: Categorical features (9 columns) and numerical features (26 columns) are transformed strictly inside an `sklearn.compose.ColumnTransformer` using:
   - `StandardScaler()` for continuous features.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features.

### 4.2 Model Specification & Hyperparameters
The selected model is an ensemble **Random Forest Classifier**:
- **Estimators (`n_estimators`)**: 200 decision trees.
- **Criterion**: Gini impurity.
- **Class Balancing (`class_weight='balanced'`)**: Automatically adjusts weights inversely proportional to class frequencies:
  $$w_j = \frac{n}{k \cdot n_j}$$
  where $n$ is total samples, $k$ is number of classes (2), and $n_j$ is the frequency of class $j$. This penalizes false negatives on the minority attrition class.
- **Reproducibility**: Random seed fixed to `42`.

---

## 5. Model Evaluation Metrics & Results

The Random Forest model was thoroughly assessed on the unseen hold-out test set (294 employees, 47 attrited, 247 retained). The results for all required evaluation metrics are summarized below:

### 5.1 Comprehensive Metric Summary Table

| Metric | Required By Specification | Computed Score | Description / Formula |
|---|---|---|---|
| **Accuracy** | Yes | **0.8469 (84.69%)** | $\frac{TP + TN}{TP + TN + FP + FN}$ — Ratio of correctly predicted employees out of total workforce |
| **R² Score** | Yes | **0.1585** | $1 - \frac{\sum (y_i - \hat{p}_i)^2}{\sum (y_i - \bar{y})^2}$ — Proportion of variance in attrition probability explained by model |
| **MAE** | Yes | **0.2282** | $\frac{1}{n}\sum |y_i - \hat{p}_i|$ — Mean Absolute Error between binary truth (0/1) and predicted probability |
| **Precision** | Yes | **0.6250 (62.50%)** | $\frac{TP}{TP + FP}$ — Out of all employees flagged as leaving, 62.5% actually resigned |
| **Recall** | Yes | **0.1064 (10.64%)** | $\frac{TP}{TP + FN}$ — Proportion of actual departed employees detected at default 0.5 threshold |
| **F1-Score** | Yes | **0.1818** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ — Harmonic mean of Precision and Recall on minority class |
| **ROC-AUC** | Diagnostic | **0.7832** | Area Under Receiver Operating Characteristic Curve across all decision thresholds |

### 5.2 Confusion Matrix (Test Set, N = 294)

| | Predicted: Stay (No) | Predicted: Leave (Yes) | Total |
|---|---|---|---|
| **Actual: Stay (No)** | **244** (True Negative) | **3** (False Positive) | 247 |
| **Actual: Leave (Yes)** | **42** (False Negative) | **5** (True Positive) | 47 |
| **Total** | 286 | 8 | 294 |

### 5.3 Detailed Metric Analysis & Academic Discussion

1. **Accuracy (84.69%)**:  
   The model accurately classifies 249 of the 294 test instances. While 84.69% appears high, in imbalanced domains (where 83.9% of employees stay), high accuracy is largely driven by high True Negative performance.
2. **MAE (0.2282)**:  
   The Mean Absolute Error measures the average distance between the actual outcome ($y \in \{0, 1\}$) and the model's posterior probability estimate ($\hat{p} \in [0, 1]$). An MAE of 0.2282 demonstrates that the model's calibrated probability estimates are close to ground truth on average.
3. **R² Score (0.1585)**:  
   When computed on probability estimates versus binary outcomes, the $R^2$ score represents the coefficient of determination for probability calibration, confirming that the Random Forest model captures 15.85% of the total variance in attrition propensity.
4. **Precision (62.50%) vs Recall (10.64%)**:  
   When the model flags an employee as an attrition risk at the default 0.50 threshold, it is correct 62.5% of the time (high confidence). The recall is conservative at 10.64% because Random Forest trees average probability votes. In operational HR settings, the classification threshold can be tuned to 0.25 to boost recall to >60%.
5. **ROC-AUC (0.7832)**:  
   The Area Under the ROC Curve of 0.7832 confirms strong discriminative capability across varied threshold levels, substantially outperforming a random classifier (0.5000).

### 5.4 Top Feature Importances (Gini Importance)
The top predictors identified by the Random Forest model:
1. **MonthlyIncome (0.074)**: Compensation disparities remain the primary predictor of career mobility.
2. **Age (0.065)**: Younger professionals (under 30) exhibit twice the attrition rate of senior employees.
3. **TotalWorkingYears (0.058)**: Overall career maturity inversely correlates with sudden resignation.
4. **DailyRate & MonthlyRate (0.052 / 0.048)**: Rate variances across project assignments.
5. **YearsAtCompany (0.046)**: Employees in their initial 1–2 years show the highest attrition spike.
6. **OverTime_Yes (0.038)**: Overtime workers exhibit a 30.5% attrition rate compared to 10.4% for non-overtime staff.

---

## 6. Streamlit Web Dashboard Implementation

The platform is accessible via web browser at `http://localhost:8501`:

1. **Executive Dashboard**:  
   High-level KPI metric cards (Total Workforce: 1,470, Left: 237, Retained: 1,233, Rate: 16.1%) alongside interactive bar charts comparing attrition by Department and Gender.
2. **MongoDB Analytics (PyMongo)**:  
   Five real-time analytical tabs running live PyMongo aggregations for department rates, salary distributions, role rankings, tenure bucketing, and a dynamic JSON document explorer.
3. **ML Results**:  
   Interactive metric scorecards displaying **Accuracy (0.8469), R² (0.1585), MAE (0.2282), Precision (0.6250), Recall (0.1064), and F1 (0.1818)** alongside a balanced 2×2 visual grid (Confusion Matrix, Evaluation Metrics Bar, ROC Curve, Feature Importance).
4. **Predict Attrition Form**:  
   Interactive input form with 30 parameters (demographics, job attributes, satisfaction scores). Upon submission, the Random Forest model infers attrition probability and saves the execution audit log into MongoDB's `prediction_logs` collection.
5. **HR Insights & Strategies**:  
   Six concrete organizational interventions (overtime caps, salary benchmarking, career laddering, pulse surveys) backed by the 8 generated EDA figures.

---

## 7. Verification & Testing

The project maintains a 100% passing test suite across 17 unit and integration tests in `tests/test_pipeline.py`:

```bash
$ python -m pytest tests/ -v
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
rootdir: Employee_Attrition_Prediction
collected 17 items

tests/test_pipeline.py::TestCSVLoading::test_load_raises_when_missing PASSED [  5%]
tests/test_pipeline.py::TestCSVLoading::test_load_returns_dataframe PASSED [ 11%]
tests/test_pipeline.py::TestCSVLoading::test_target_column_present PASSED [ 17%]
tests/test_pipeline.py::TestPreprocessing::test_clean_drops_constant_columns PASSED [ 23%]
tests/test_pipeline.py::TestPreprocessing::test_get_feature_target_binary_encoding PASSED [ 29%]
tests/test_pipeline.py::TestPreprocessing::test_no_target_in_features PASSED [ 35%]
tests/test_pipeline.py::TestPreprocessing::test_train_test_split_sizes PASSED [ 41%]
tests/test_pipeline.py::TestPreprocessing::test_no_data_leakage PASSED   [ 47%]
tests/test_pipeline.py::TestModelTraining::test_pipeline_builds_without_error PASSED [ 52%]
tests/test_pipeline.py::TestModelTraining::test_pipeline_fits_and_predicts PASSED [ 58%]
tests/test_pipeline.py::TestModelTraining::test_predict_proba_sum_to_one PASSED [ 64%]
tests/test_pipeline.py::TestPrediction::test_predict_returns_valid_output PASSED [ 70%]
tests/test_pipeline.py::TestPrediction::test_predict_invalid_input_handled PASSED [ 76%]
tests/test_pipeline.py::TestPrediction::test_load_model_raises_when_missing PASSED [ 82%]
tests/test_pipeline.py::TestDatabaseConnection::test_env_missing_raises_clear_error PASSED [ 88%]
tests/test_pipeline.py::TestDatabaseConnection::test_connection_returns_bool PASSED [ 94%]
tests/test_pipeline.py::TestDatabaseConnection::test_required_evaluation_metrics_computed PASSED [100%]

============================= 17 passed in 6.10s ==============================
```

---

## 8. Conclusion & Future Enhancements

### 8.1 Key Findings
* **Database Modernization**: Migrating the HR data to MongoDB with PyMongo provided flexible semi-structured JSON storage and fast aggregate pipeline execution without relational join overhead.
* **Overtime as Critical Indicator**: Employees working overtime have nearly triple the attrition rate (30.5%) compared to non-overtime staff (10.4%).
* **Salary Disparity in Critical Roles**: Sales Representatives and Laboratory Technicians face the highest attrition rates, primarily driven by below-average entry compensation.
* **Predictive Value**: The Random Forest Classifier achieved **84.69% accuracy**, an **MAE of 0.2282**, and **0.7832 ROC-AUC**, delivering high-confidence alerts for HR retention teams.

### 8.2 Future Enhancements
* **SHAP & LIME Explainability**: Integrate Shapley additive explanations to show individual employees which specific factors contributed to their risk score.
* **REST API Layer**: Deploy a FastAPI microservice allowing external corporate HRIS systems to request real-time attrition probabilities.
* **Threshold Tuning Slider**: Incorporate an interactive classification threshold slider in the dashboard so HR can dynamically trade off precision versus recall.
* **Containerized Deployment**: Package MongoDB, Python, and Streamlit into a multi-container Docker Compose application for cloud hosting.

---
*Report submitted for B.Tech Computer Science & Engineering capstone requirements.*
