"""
evaluate_model.py
-----------------
Evaluates the Random Forest Classifier on the held-out test set.
Computes all required academic metrics:
  - Accuracy
  - R² Score (calibration of probability estimates)
  - MAE (Mean Absolute Error on predicted probabilities)
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix
Saves evaluation charts to visualizations/ and a JSON report to reports/.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
    mean_absolute_error, r2_score,
)

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR    = os.path.join(BASE_DIR, "visualizations")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(VIZ_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

ACCENT_COLOR = "#1D3557"
BG_COLOR     = "#F1FAEE"


def _save(filename: str) -> str:
    path = os.path.join(VIZ_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"   💾 Saved: {filename}")
    return path


# ─────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str = "random_forest") -> dict:
    """
    Evaluate a fitted pipeline and return a comprehensive metrics dict.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    mae_val = round(float(mean_absolute_error(y_test, y_prob)), 4)
    r2_val  = round(float(r2_score(y_test, y_prob)), 4)

    metrics = {
        "model_name" : model_name,
        "accuracy"   : round(float(accuracy_score(y_test, y_pred)), 4),
        "r2_score"   : r2_val,
        "mae"        : mae_val,
        "precision"  : round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall"     : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score"   : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc"    : round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["No", "Yes"]),
    }

    print(f"\n{'='*55}")
    print(f"  MODEL EVALUATION: {model_name.replace('_', ' ').title()}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  R² Score  : {metrics['r2_score']:.4f}")
    print(f"  MAE       : {metrics['mae']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (class=Yes)")
    print(f"  Recall    : {metrics['recall']:.4f}  (class=Yes)")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}  (class=Yes)")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n{metrics['classification_report']}")

    return metrics


# ─────────────────────────────────────────────────────────────────
# Confusion matrix chart
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(metrics: dict) -> str:
    """Plot confusion matrix for the Random Forest model."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    cm = np.array(metrics["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay (No)", "Leave (Yes)"],
                yticklabels=["Stay (No)", "Leave (Yes)"],
                ax=ax, linewidths=1.5, linecolor="white",
                annot_kws={"size": 15, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {metrics['model_name'].replace('_', ' ').title()}",
                 color=ACCENT_COLOR, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    return _save("09_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────
# Model Metrics Bar Chart
# ─────────────────────────────────────────────────────────────────
def plot_metrics_bar(metrics: dict) -> str:
    """Plot bar chart of all required evaluation metrics."""
    metric_keys   = ["accuracy", "r2_score", "mae", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "R² Score", "MAE", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    values        = [metrics[k] for k in metric_keys]

    colors = ["#2A9D8F", "#457B9D", "#E76F51", "#F4A261", "#E63946", "#1D3557", "#264653"]
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)

    bars = ax.bar(metric_labels, values, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:.4f}", ha="center", va=va, fontsize=10, fontweight="bold")

    ax.set_title(f"Evaluation Metrics — {metrics['model_name'].replace('_', ' ').title()}",
                 fontsize=14, fontweight="bold", color=ACCENT_COLOR, pad=12)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.axhline(y=0.0, color="gray", linestyle="-", linewidth=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    return _save("10_model_comparison.png")


# ─────────────────────────────────────────────────────────────────
# ROC curve
# ─────────────────────────────────────────────────────────────────
def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """Plot ROC curve for the Random Forest classifier."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.4f})",
            color="#1D3557", linewidth=2.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline (AUC = 0.5000)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#457B9D")

    ax.set_title("Receiver Operating Characteristic (ROC) Curve",
                 fontsize=14, fontweight="bold", color=ACCENT_COLOR, pad=12)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.legend(loc="lower right")

    return _save("11_roc_curves.png")


# ─────────────────────────────────────────────────────────────────
# Feature importance chart (RF)
# ─────────────────────────────────────────────────────────────────
def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 20) -> str:
    """Plot top N feature importances from the Random Forest model."""
    top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color="#457B9D", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=14, fontweight="bold", color=ACCENT_COLOR, pad=12)
    ax.set_xlabel("Gini Importance Score", fontweight="bold")
    return _save("12_feature_importance.png")


# ─────────────────────────────────────────────────────────────────
# Save evaluation report
# ─────────────────────────────────────────────────────────────────
def save_evaluation_report(metrics: dict, best_model_name: str = "random_forest") -> str:
    """Save evaluation results as JSON for consumption by the dashboard."""
    report = {
        "models"    : [metrics],
        "best_model": best_model_name,
        "primary_model": metrics,
    }
    path = os.path.join(REPORT_DIR, "model_evaluation.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   💾 Evaluation report saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import run_preprocessing
    from train_model import load_model, get_feature_importances

    (X_train, X_test, y_train, y_test,
     numeric_cols, categorical_cols, df_clean) = run_preprocessing()

    rf_model = load_model("random_forest")
    metrics_rf = evaluate_model(rf_model, X_test, y_test, "random_forest")

    print("\n📊 Generating evaluation charts …")
    plot_confusion_matrix(metrics_rf)
    plot_metrics_bar(metrics_rf)
    plot_roc_curve(rf_model, X_test, y_test)

    fi_df = get_feature_importances(rf_model, numeric_cols, categorical_cols)
    plot_feature_importance(fi_df)

    save_evaluation_report(metrics_rf, "random_forest")
    print("\n✅ Evaluation complete. All charts & reports saved.\n")
