"""
train_model.py
==============
Trains and evaluates the workload model for Tawazon.

Steps:
    1. Load employees.csv and drop non-feature columns.
    2. Evaluate the model on the ORIGINAL (imbalanced) data.
    3. Balance the data (400 burnout + 400 no-burnout) and evaluate again.
    4. Compare both results (Accuracy, Precision, Recall, F1, Balanced Accuracy)
       and save confusion matrices for each.
    5. Save the balanced Random Forest as the final model.

Run:
    python train_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(ROOT, "data", "employees.csv")
BALANCED_CSV = os.path.join(ROOT, "data", "employees_balanced.csv")
MODEL_DIR    = os.path.join(ROOT, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "workload_model.pkl")
RESULTS_CSV  = os.path.join(MODEL_DIR, "model_comparison.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────
SEED          = 42
TEST_SIZE     = 0.20          # 80% train / 20% test
PER_CLASS     = 400           # balanced target: 400 burnout + 400 no-burnout
FEATURES      = [
    "avg_daily_hours",
    "overtime_hours_week",
    "absence_days_month",
    "late_arrivals_month",
    "tasks_assigned",
    "vacation_days_last_year",
]
TARGET        = "burnout"
LABELS        = ["No Burnout", "Burnout"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Remove identifiers and sensitive attributes (not used for prediction)
    df = df.drop(columns=["employee_id", "name", "gender", "age"], errors="ignore")
    print(f"[Data] Loaded {len(df)} records")
    print(f"[Data] Class counts:\n{df[TARGET].value_counts().to_string()}\n")
    return df


def balance_data(df):
    """Undersample the majority class so both classes have PER_CLASS rows."""
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0]
    if len(pos) < PER_CLASS or len(neg) < PER_CLASS:
        raise ValueError("Not enough rows in one class to reach the balanced target.")
    balanced = pd.concat([
        pos.sample(n=PER_CLASS, random_state=SEED),
        neg.sample(n=PER_CLASS, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    balanced.to_csv(BALANCED_CSV, index=False)
    print(f"[Balance] Saved balanced data -> {BALANCED_CSV}")
    print(f"[Balance] Class counts:\n{balanced[TARGET].value_counts().to_string()}\n")
    return balanced


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=SEED)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=SEED, n_jobs=-1
        ),
    }


def evaluate(df, dataset_name):
    """Split 80/20 (stratified), train both models, return metrics and predictions."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"[Split: {dataset_name}] Train: {len(X_train)} | Test: {len(X_test)}")

    rows, fitted, preds = [], {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted[name], preds[name] = model, (y_test, y_pred)

        print(f"\n--- {name} | {dataset_name} ---")
        print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

        rows.append({
            "Dataset": dataset_name,
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 1),
            "Balanced Accuracy (%)": round(balanced_accuracy_score(y_test, y_pred) * 100, 1),
            "Precision (%)": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
            "Recall (%)": round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
            "F1 (%)": round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
        })
    return rows, fitted, preds


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Saved] {path}")


def main():
    df = load_data()

    # 1) Original data
    rows_before, _, preds_before = evaluate(df, "Before balancing")

    # 2) Balanced data
    balanced = balance_data(df)
    rows_after, models_after, preds_after = evaluate(balanced, "After balancing")

    # 3) Comparison table
    results = pd.DataFrame(rows_before + rows_after)
    results.to_csv(RESULTS_CSV, index=False)
    print("\n=== Comparison (Before vs After balancing) ===")
    print(results.to_string(index=False))
    print(f"\n[Saved] {RESULTS_CSV}")

    # 4) Confusion matrices (Random Forest, before and after)
    y_t, y_p = preds_before["Random Forest"]
    save_confusion_matrix(y_t, y_p, "Random Forest - Before Balancing", "confusion_matrix_before.png")
    y_t, y_p = preds_after["Random Forest"]
    save_confusion_matrix(y_t, y_p, "Random Forest - After Balancing", "confusion_matrix_after.png")

    # 5) Save final model: balanced Random Forest
    final_model = models_after["Random Forest"]
    joblib.dump(final_model, MODEL_PATH)
    print(f"[Saved] Final model (balanced Random Forest) -> {MODEL_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
