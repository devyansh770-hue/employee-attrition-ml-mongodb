"""
database/insert_data.py
-----------------------
Idempotent CSV-to-MongoDB import script using PyMongo.
- Reads the IBM HR Analytics CSV dataset
- Structures documents into clean MongoDB collections:
    1. 'employees': Core collection with 1,470 employee profiles
    2. 'departments': Summary and lookup metadata
    3. 'job_roles': Summary and lookup metadata
- Creates efficient indexes on key query fields (emp_number, department, attrition, etc.)
- Uses bulk_write with ReplaceOne(upsert=True) so the script is 100% idempotent
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from pymongo import ReplaceOne, ASCENDING

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from database_connection import get_mongo_db
from data_preprocessing import load_dataset, CSV_PATH


def create_indexes(db):
    """Create indexes for high-performance MongoDB queries."""
    print("⚡ Creating MongoDB indexes …")
    employees = db["employees"]
    employees.create_index([("emp_number", ASCENDING)], unique=True)
    employees.create_index([("department", ASCENDING)])
    employees.create_index([("job_role", ASCENDING)])
    employees.create_index([("attrition", ASCENDING)])
    employees.create_index([("monthly_income", ASCENDING)])
    employees.create_index([("years_at_company", ASCENDING)])
    employees.create_index([("age", ASCENDING)])
    print("   ✅ Indexes established.")


def import_employees(db, df: pd.DataFrame):
    """
    Transform DataFrame rows into structured JSON documents and
    bulk upsert into the 'employees' MongoDB collection.
    """
    print(f"📥 Preparing {len(df):,} employee documents for MongoDB …")
    operations = []

    for _, row in df.iterrows():
        emp_number = int(row["EmployeeNumber"])
        doc = {
            "emp_number"      : emp_number,
            "age"             : int(row["Age"]),
            "gender"          : str(row["Gender"]),
            "marital_status"  : str(row["MaritalStatus"]),
            "department"      : str(row["Department"]),
            "job_role"        : str(row["JobRole"]),
            "job_level"       : int(row["JobLevel"]),
            "business_travel" : str(row["BusinessTravel"]),
            "over_time"       : str(row["OverTime"]),
            "monthly_income"  : float(row["MonthlyIncome"]),
            "daily_rate"      : float(row["DailyRate"]),
            "hourly_rate"     : float(row["HourlyRate"]),
            "monthly_rate"    : float(row["MonthlyRate"]),
            "percent_salary_hike": int(row["PercentSalaryHike"]),
            "stock_option_level" : int(row["StockOptionLevel"]),
            "education"       : int(row["Education"]),
            "education_field" : str(row["EducationField"]),
            "environment_satisfaction"  : int(row["EnvironmentSatisfaction"]),
            "job_involvement"           : int(row["JobInvolvement"]),
            "job_satisfaction"          : int(row["JobSatisfaction"]),
            "performance_rating"        : int(row["PerformanceRating"]),
            "relationship_satisfaction" : int(row["RelationshipSatisfaction"]),
            "work_life_balance"         : int(row["WorkLifeBalance"]),
            "total_working_years"       : int(row["TotalWorkingYears"]),
            "years_at_company"          : int(row["YearsAtCompany"]),
            "years_in_current_role"     : int(row["YearsInCurrentRole"]),
            "years_since_last_promotion": int(row["YearsSinceLastPromotion"]),
            "years_with_curr_manager"   : int(row["YearsWithCurrManager"]),
            "num_companies_worked"      : int(row["NumCompaniesWorked"]),
            "training_times_last_year"  : int(row["TrainingTimesLastYear"]),
            "distance_from_home"        : int(row["DistanceFromHome"]),
            "attrition"                 : str(row["Attrition"]),
            "attrition_binary"          : 1 if str(row["Attrition"]) == "Yes" else 0,
        }
        operations.append(
            ReplaceOne({"emp_number": emp_number}, doc, upsert=True)
        )

    result = db["employees"].bulk_write(operations)
    print(f"   ✅ 'employees' collection: matched={result.matched_count}, upserted={len(result.upserted_ids)}")


def populate_lookups(db, df: pd.DataFrame):
    """Compute and store aggregate summary collections for departments and roles."""
    print("📊 Generating lookup collections …")

    # Departments lookup
    dept_col = db["departments"]
    dept_summary = (
        df.groupby("Department")
        .agg(
            total_employees=("Attrition", "count"),
            attrited=("Attrition", lambda x: (x == "Yes").sum()),
            avg_income=("MonthlyIncome", "mean"),
        )
        .reset_index()
    )
    dept_summary["attrition_rate_pct"] = (
        dept_summary["attrited"] / dept_summary["total_employees"] * 100
    ).round(2)
    dept_summary["avg_income"] = dept_summary["avg_income"].round(2)

    dept_ops = [
        ReplaceOne(
            {"department_name": r["Department"]},
            {
                "department_name": r["Department"],
                "total_employees": int(r["total_employees"]),
                "attrited": int(r["attrited"]),
                "attrition_rate_pct": float(r["attrition_rate_pct"]),
                "avg_income": float(r["avg_income"]),
            },
            upsert=True,
        )
        for _, r in dept_summary.iterrows()
    ]
    dept_col.bulk_write(dept_ops)

    # Job roles lookup
    role_col = db["job_roles"]
    role_summary = (
        df.groupby("JobRole")
        .agg(
            total_employees=("Attrition", "count"),
            attrited=("Attrition", lambda x: (x == "Yes").sum()),
            avg_income=("MonthlyIncome", "mean"),
        )
        .reset_index()
    )
    role_summary["attrition_rate_pct"] = (
        role_summary["attrited"] / role_summary["total_employees"] * 100
    ).round(2)
    role_summary["avg_income"] = role_summary["avg_income"].round(2)

    role_ops = [
        ReplaceOne(
            {"job_role_name": r["JobRole"]},
            {
                "job_role_name": r["JobRole"],
                "total_employees": int(r["total_employees"]),
                "attrited": int(r["attrited"]),
                "attrition_rate_pct": float(r["attrition_rate_pct"]),
                "avg_income": float(r["avg_income"]),
            },
            upsert=True,
        )
        for _, r in role_summary.iterrows()
    ]
    role_col.bulk_write(role_ops)

    print("   ✅ Lookup collections populated.")


def main():
    print("\n" + "=" * 60)
    print("MONGODB DATA IMPORT (PyMongo)")
    print("=" * 60)

    db = get_mongo_db()
    df = load_dataset(CSV_PATH)
    create_indexes(db)
    import_employees(db, df)
    populate_lookups(db, df)

    count = db["employees"].count_documents({})
    print(f"\n🎉 MongoDB import complete. Total documents in 'employees': {count:,}\n")


if __name__ == "__main__":
    main()
