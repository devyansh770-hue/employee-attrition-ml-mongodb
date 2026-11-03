"""
data_analysis.py
----------------
Performs EDA and generates all visualizations saved to the
visualizations/ folder. Also handles CSV-to-MySQL data import.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR      = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Colour palette
PALETTE      = {"Yes": "#E63946", "No": "#457B9D"}
ACCENT_COLOR = "#1D3557"
BG_COLOR     = "#F1FAEE"

# ─────────────────────────────────────────────────────────────────
# Helper: save figure
# ─────────────────────────────────────────────────────────────────
def _save(filename: str) -> str:
    """Save the current matplotlib figure and return the full path."""
    path = os.path.join(VIZ_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"   💾 Saved: {filename}")
    return path


# ─────────────────────────────────────────────────────────────────
# 1. Attrition distribution (pie + bar)
# ─────────────────────────────────────────────────────────────────
def plot_attrition_distribution(df: pd.DataFrame) -> str:
    print("📊 Plotting attrition distribution …")
    counts = df["Attrition"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)
    fig.suptitle("Employee Attrition Distribution", fontsize=16, fontweight="bold", color=ACCENT_COLOR)

    # Bar
    bars = axes[0].bar(counts.index, counts.values,
                       color=[PALETTE[k] for k in counts.index], edgecolor="white", linewidth=1.5)
    axes[0].set_title("Count", color=ACCENT_COLOR)
    axes[0].set_ylabel("Number of Employees")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha="center", fontweight="bold")

    # Pie
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[PALETTE[k] for k in counts.index],
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Proportion", color=ACCENT_COLOR)

    return _save("01_attrition_distribution.png")


# ─────────────────────────────────────────────────────────────────
# 2. Department-wise attrition
# ─────────────────────────────────────────────────────────────────
def plot_department_attrition(df: pd.DataFrame) -> str:
    print("📊 Plotting department-wise attrition …")
    dept_attr = (
        df.groupby(["Department", "Attrition"])
        .size().reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    sns.barplot(data=dept_attr, x="Department", y="count", hue="Attrition",
                palette=PALETTE, ax=ax, edgecolor="white")
    ax.set_title("Department-wise Attrition", fontsize=14, fontweight="bold", color=ACCENT_COLOR)
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Employees")
    ax.legend(title="Attrition")
    return _save("02_department_attrition.png")


# ─────────────────────────────────────────────────────────────────
# 3. Job-role-wise attrition (horizontal bar)
# ─────────────────────────────────────────────────────────────────
def plot_jobrole_attrition(df: pd.DataFrame) -> str:
    print("📊 Plotting job-role-wise attrition …")
    rate = (
        df.groupby("JobRole")["Attrition"]
        .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
        .sort_values(ascending=True)
        .reset_index()
    )
    rate.columns = ["JobRole", "Attrition_Rate"]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    bars = ax.barh(rate["JobRole"], rate["Attrition_Rate"],
                   color=PALETTE["Yes"], edgecolor="white")
    ax.set_title("Job Role – Attrition Rate (%)", fontsize=14, fontweight="bold", color=ACCENT_COLOR)
    ax.set_xlabel("Attrition Rate (%)")
    for bar, val in zip(bars, rate["Attrition_Rate"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    return _save("03_jobrole_attrition.png")


# ─────────────────────────────────────────────────────────────────
# 4. Monthly income distribution
# ─────────────────────────────────────────────────────────────────
def plot_income_distribution(df: pd.DataFrame) -> str:
    print("📊 Plotting monthly income distribution …")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    for label, grp in df.groupby("Attrition"):
        grp["MonthlyIncome"].plot.kde(ax=ax, label=label, color=PALETTE[label], linewidth=2.5)
    ax.set_title("Monthly Income Distribution by Attrition", fontsize=14, fontweight="bold", color=ACCENT_COLOR)
    ax.set_xlabel("Monthly Income ($)")
    ax.legend(title="Attrition")
    ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), alpha=0.1, color=PALETTE["Yes"])
    ax.fill_between(ax.lines[1].get_xdata(), ax.lines[1].get_ydata(), alpha=0.1, color=PALETTE["No"])
    return _save("04_income_distribution.png")


# ─────────────────────────────────────────────────────────────────
# 5. Age distribution
# ─────────────────────────────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame) -> str:
    print("📊 Plotting age distribution …")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    for label, grp in df.groupby("Attrition"):
        sns.histplot(grp["Age"], kde=True, ax=ax, label=label,
                     color=PALETTE[label], alpha=0.5, bins=20)
    ax.set_title("Age Distribution by Attrition", fontsize=14, fontweight="bold", color=ACCENT_COLOR)
    ax.set_xlabel("Age")
    ax.legend(title="Attrition")
    return _save("05_age_distribution.png")


# ─────────────────────────────────────────────────────────────────
# 6. Overtime vs attrition
# ─────────────────────────────────────────────────────────────────
def plot_overtime_attrition(df: pd.DataFrame) -> str:
    print("📊 Plotting overtime vs attrition …")
    ot = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_COLOR)
    sns.barplot(data=ot, x="OverTime", y="count", hue="Attrition",
                palette=PALETTE, ax=ax, edgecolor="white")
    ax.set_title("Overtime vs Attrition", fontsize=14, fontweight="bold", color=ACCENT_COLOR)
    ax.set_xlabel("Overtime")
    ax.set_ylabel("Count")
    return _save("06_overtime_attrition.png")


# ─────────────────────────────────────────────────────────────────
# 7. Job satisfaction analysis
# ─────────────────────────────────────────────────────────────────
def plot_job_satisfaction(df: pd.DataFrame) -> str:
    print("📊 Plotting job satisfaction …")
    sat = df.groupby(["JobSatisfaction", "Attrition"]).size().reset_index(name="count")
    sat["JobSatisfaction"] = sat["JobSatisfaction"].map(
        {1: "Low", 2: "Medium", 3: "High", 4: "Very High"})
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG_COLOR)
    sns.barplot(data=sat, x="JobSatisfaction", y="count", hue="Attrition",
                palette=PALETTE, ax=ax, edgecolor="white",
                order=["Low", "Medium", "High", "Very High"])
    ax.set_title("Job Satisfaction vs Attrition", fontsize=14, fontweight="bold", color=ACCENT_COLOR)
    ax.set_xlabel("Job Satisfaction Level")
    ax.set_ylabel("Count")
    return _save("07_job_satisfaction.png")


# ─────────────────────────────────────────────────────────────────
# 8. Correlation heatmap (numeric features only)
# ─────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    print("📊 Plotting correlation heatmap …")
    df_num = df.copy()
    df_num["Attrition"] = df_num["Attrition"].map({"Yes": 1, "No": 0})
    # Keep numeric only
    df_num = df_num.select_dtypes(include=np.number)
    corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(18, 14), facecolor=BG_COLOR)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 7},
                vmin=-1, vmax=1, center=0)
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold", color=ACCENT_COLOR)
    return _save("08_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────
def run_all_visualizations(df: pd.DataFrame) -> list:
    """
    Generate all 8 EDA charts and return a list of saved file paths.
    """
    print("\n" + "="*60)
    print("GENERATING EDA VISUALIZATIONS")
    print("="*60)
    paths = [
        plot_attrition_distribution(df),
        plot_department_attrition(df),
        plot_jobrole_attrition(df),
        plot_income_distribution(df),
        plot_age_distribution(df),
        plot_overtime_attrition(df),
        plot_job_satisfaction(df),
        plot_correlation_heatmap(df),
    ]
    print(f"\n✅ {len(paths)} visualizations saved to: {VIZ_DIR}\n")
    return paths


if __name__ == "__main__":
    from data_preprocessing import load_dataset
    df = load_dataset()
    run_all_visualizations(df)
