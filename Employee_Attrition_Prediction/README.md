# Employee Attrition Prediction & HR Analytics System (MongoDB & PyMongo)

> **Academic Project | B.Tech Computer Science & Engineering**  
> Uses IBM HR Analytics dataset, **MongoDB with PyMongo**, Scikit-learn **Random Forest Classifier**, and a Streamlit dashboard.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Technology Stack](#technology-stack)
5. [System Architecture](#system-architecture)
6. [Project Structure](#project-structure)
7. [MongoDB Document Schema](#mongodb-document-schema)
8. [Installation & Setup](#installation--setup)
9. [MongoDB Data Ingestion](#mongodb-data-ingestion)
10. [Model Training & Required Evaluation Metrics](#model-training--required-evaluation-metrics)
11. [Dashboard Execution](#dashboard-execution)
12. [Running Tests](#running-tests)
13. [Limitations & Future Enhancements](#limitations--future-enhancements)

---

## Introduction

Employee attrition costs organizations an average of **6–9 months' salary** per departed employee in recruitment, onboarding, and training costs. Identifying employees at risk of leaving allows HR departments to implement targeted retention strategies.

This project delivers an end-to-end enterprise solution that:
- Stores and queries workforce data in **MongoDB** using **PyMongo** document modeling and multi-stage aggregation pipelines
- Implements an ensemble **Random Forest Machine Learning Classifier** to predict attrition likelihood
- Computes comprehensive evaluation metrics: **Accuracy, R² Score, MAE, Precision, Recall, and F1-Score**
- Delivers an interactive **Streamlit dashboard** with live MongoDB analytics, model diagnostics, and prediction logging

---

## Technology Stack

| Component | Technology |
|---|---|
| **Database** | MongoDB Server (NoSQL Document Store) |
| **Database Connector** | PyMongo (>= 4.8.0) |
| **Language** | Python 3.10+ (tested on Python 3.13) |
| **Machine Learning** | Scikit-learn (Random Forest Classifier Pipeline) |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Web Dashboard** | Streamlit |
| **Model Persistence** | Joblib |
| **Testing** | Pytest |
| **Configuration** | python-dotenv |

---

## Project Structure

```
Employee_Attrition_Prediction/
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← 1,470 records × 35 columns
├── database/
│   └── insert_data.py       ← PyMongo idempotent CSV → MongoDB importer
├── src/
│   ├── database_connection.py   ← PyMongo connection factory & health check
│   ├── data_preprocessing.py    ← Preprocessing, cleaning, stratified split
│   ├── data_analysis.py         ← EDA + 8 automated visualisations
│   ├── train_model.py           ← Random Forest classifier pipeline
│   ├── evaluate_model.py        ← Metrics (Acc, R², MAE, Precision, Recall, F1)
│   └── prediction.py            ← Inference function with MongoDB logging
├── models/
│   └── random_forest.joblib     ← Trained serialized model
├── visualizations/              ← 12 auto-generated visual charts
├── reports/
│   ├── model_evaluation.json    ← Evaluation report JSON
│   └── project_report.md        ← Full B.Tech Academic Project Report
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py         ← 17 passing Pytest unit tests
├── app.py                       ← Streamlit dashboard with MongoDB Analytics
├── requirements.txt
├── .env.example
├── .env
└── README.md
```

---

## MongoDB Document Schema

Documents in the `employees` collection are structured as follows:

```json
{
  "_id": ObjectId("..."),
  "emp_number": 101,
  "age": 35,
  "gender": "Male",
  "marital_status": "Single",
  "department": "Research & Development",
  "job_role": "Research Scientist",
  "job_level": 2,
  "monthly_income": 5500.0,
  "over_time": "No",
  "years_at_company": 5,
  "total_working_years": 10,
  "attrition": "No",
  "attrition_binary": 0
}
```

Indexed fields: `emp_number` (unique), `department`, `job_role`, `attrition`, `monthly_income`, `years_at_company`, `age`.

---

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure MongoDB
Ensure MongoDB is running (default `localhost:27017`). The `.env` file contains:
```env
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=employee_attrition_db
```

### 3. Ingest Data into MongoDB
```bash
python database/insert_data.py
```
This populates 1,470 employee records and establishes lookup collections and indexes.

---

## Model Training & Required Evaluation Metrics

Run the machine learning pipeline:
```bash
# Step 1: Generate EDA Visualizations
python src/data_analysis.py

# Step 2: Train Random Forest Classifier
python src/train_model.py

# Step 3: Evaluate on Hold-Out Test Set
python src/evaluate_model.py
```

### Results on Test Set (N = 294):

| Metric | Score | Purpose |
|---|---|---|
| **Accuracy** | **84.69%** | Overall correct classification rate |
| **R² Score** | **0.1585** | Probability estimate variance explained |
| **MAE** | **0.2282** | Mean Absolute Error between truth (0/1) and probability |
| **Precision** | **62.50%** | Reliability when predicting attrition (Yes) |
| **Recall** | **10.64%** | Detection rate at default 0.50 threshold |
| **F1-Score** | **0.1818** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | **0.7832** | Area under the ROC curve |

---

## Dashboard Execution

Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Opens in your browser at: **`http://localhost:8501`**

### Dashboard Sections:
1. **📊 Executive Dashboard**: Total headcount, turnover rates, department and gender charts.
2. **🍃 MongoDB Analytics**: Live PyMongo aggregation pipelines for department turnover, salary statistics, tenure buckets, and direct JSON document inspector.
3. **🤖 ML Results**: Displays all 6 required academic metrics, confusion matrix, ROC curve, and feature importances.
4. **🔮 Predict Attrition**: Form to test hypothetical employee profiles with instant probabilities and automatic prediction logging to MongoDB.
5. **💡 HR Insights**: Data-driven retention recommendations and 8 EDA charts.

---

## Running Tests

Execute the automated test suite with pytest:
```bash
python -m pytest tests/ -v
```
All **17 tests pass (100% pass rate)**.
