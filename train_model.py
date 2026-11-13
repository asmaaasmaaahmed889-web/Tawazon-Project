"""
train_model.py
==============
Trains and evaluates the Workload Burnout Prediction model according to
the supervisor Dr. Amir Al-Attar's directive:
1. Trains on Balanced Data:
   - data/train.csv (640 records: 320 Burnout, 320 No Burnout)
   - Evaluates on data/test.csv (160 records: 80 Burnout, 80 No Burnout)
2. Compares performance directly:
   - Imbalanced Model (Original 1,000 dataset: 800 train / 200 test)
   - Balanced Model (New 800 dataset: 640 train / 160 test)
3. Generates:
   - models/workload_model.pkl (Saved Balanced Random Forest model)
   - models/confusion_matrix.png
   - models/feature_importance.png
   - models/comparison_balance_vs_imbalance.png
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
IMBALANCED_PATH = os.path.join(DATA_DIR, "employees_1000_imbalanced.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "workload_model.pkl")

FEATURES = [
    "avg_daily_hours",
    "overtime_hours_week",
    "absence_days_month",
    "late_arrivals_month",
    "tasks_assigned",
    "vacation_days_last_year",
]
TARGET = "burnout"
SEED = 42


def evaluate_model_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "cm": cm,
        "report": classification_report(y_test, preds, target_names=["No Burnout", "Burnout"])
    }


def main():
    print("=" * 75)
    print("  TAWAZON — MODEL TRAINING & PERFORMANCE COMPARISON")
    print("  Directive: Dr. Amir Al-Attar (Supervisor Follow-up)")
    print("=" * 75)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. TRAIN IMBALANCED MODEL (Baseline / Before Balancing)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 1] Training on Imbalanced Dataset (1,000 employees)...")
    if os.path.exists(IMBALANCED_PATH):
        df_imbalanced = pd.read_csv(IMBALANCED_PATH)
    else:
        # Fallback if not generated yet
        from balance_dataset import balance_and_split
        balance_and_split()
        df_imbalanced = pd.read_csv(IMBALANCED_PATH)

    X_imb = df_imbalanced[FEATURES]
    y_imb = df_imbalanced[TARGET]
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.20, random_state=SEED, stratify=y_imb
    )
    
    rf_imbalanced = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )
    rf_imbalanced.fit(X_train_imb, y_train_imb)
    metrics_imb = evaluate_model_metrics(rf_imbalanced, X_test_imb, y_test_imb)
    print(f"  Imbalanced RF Test Size: {len(X_test_imb)} (No Burnout: {(y_test_imb==0).sum()}, Burnout: {(y_test_imb==1).sum()})")
    print(f"  Accuracy: {metrics_imb['Accuracy']*100:.2f}% | F1: {metrics_imb['F1-Score']*100:.2f}% | Recall: {metrics_imb['Recall (Sensitivity)']*100:.2f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. TRAIN BALANCED MODEL (After Balancing to 800 employees: 640 train / 160 test)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Step 2] Training on Balanced Dataset (800 employees: 400/400)...")
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
        from balance_dataset import balance_and_split
        balance_and_split()

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"  Train set: {len(train_df)} rows (Burnout: {(train_df[TARGET]==1).sum()}, Normal: {(train_df[TARGET]==0).sum()})")
    print(f"  Test set:  {len(test_df)} rows (Burnout: {(test_df[TARGET]==1).sum()}, Normal: {(test_df[TARGET]==0).sum()})")

    X_train_bal = train_df[FEATURES]
    y_train_bal = train_df[TARGET]
    X_test_bal = test_df[FEATURES]
    y_test_bal = test_df[TARGET]

    # Baseline on balanced: Logistic Regression
    scaler = StandardScaler()
    X_train_bal_sc = scaler.fit_transform(X_train_bal)
    X_test_bal_sc = scaler.transform(X_test_bal)

    lr_bal = LogisticRegression(max_iter=500, random_state=SEED)
    lr_bal.fit(X_train_bal_sc, y_train_bal)
    metrics_lr_bal = evaluate_model_metrics(lr_bal, X_test_bal_sc, y_test_bal)

    # Random Forest on balanced
    rf_balanced = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )
    rf_balanced.fit(X_train_bal, y_train_bal)
    metrics_bal = evaluate_model_metrics(rf_balanced, X_test_bal, y_test_bal)

    print(f"  Balanced RF Test Size: {len(X_test_bal)} (No Burnout: {(y_test_bal==0).sum()}, Burnout: {(y_test_bal==1).sum()})")
    print(f"  Accuracy: {metrics_bal['Accuracy']*100:.2f}% | F1: {metrics_bal['F1-Score']*100:.2f}% | Recall: {metrics_bal['Recall (Sensitivity)']*100:.2f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. DIRECT PERFORMANCE COMPARISON TABLE (Requested by Dr. Amir Al-Attar)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SUPERVISOR COMPARISON TABLE: [ Imbalance vs. Balance ]")
    print("=" * 75)

    comparison_df = pd.DataFrame({
        "Metric / Property": [
            "Total Dataset Size",
            "Class Ratio (0 / 1)",
            "Training Set Size",
            "Test Set Size",
            "Accuracy",
            "Precision (Burnout)",
            "Recall (Sensitivity)",
            "Specificity (Normal)",
            "F1-Score",
            "ROC-AUC",
            "True Negatives (TN)",
            "False Positives (FP)",
            "False Negatives (FN)",
            "True Positives (TP)"
        ],
        "Imbalance Model (Original)": [
            f"{len(df_imbalanced)} employees",
            f"{(y_imb==0).sum()} / {(y_imb==1).sum()} (55.9% / 44.1%)",
            f"{len(X_train_imb)} (80%)",
            f"{len(X_test_imb)} (20%)",
            f"{metrics_imb['Accuracy']*100:.2f}%",
            f"{metrics_imb['Precision']*100:.2f}%",
            f"{metrics_imb['Recall (Sensitivity)']*100:.2f}%",
            f"{metrics_imb['Specificity']*100:.2f}%",
            f"{metrics_imb['F1-Score']*100:.2f}%",
            f"{metrics_imb['ROC-AUC']*100:.2f}%",
            f"{metrics_imb['TN']}",
            f"{metrics_imb['FP']}",
            f"{metrics_imb['FN']}",
            f"{metrics_imb['TP']}"
        ],
        "Balance Model (Supervisor Directives)": [
            f"{len(train_df) + len(test_df)} employees",
            f"400 / 400 (50.0% / 50.0%)",
            f"{len(X_train_bal)} (80%)",
            f"{len(X_test_bal)} (20%)",
            f"{metrics_bal['Accuracy']*100:.2f}%",
            f"{metrics_bal['Precision']*100:.2f}%",
            f"{metrics_bal['Recall (Sensitivity)']*100:.2f}%",
            f"{metrics_bal['Specificity']*100:.2f}%",
            f"{metrics_bal['F1-Score']*100:.2f}%",
            f"{metrics_bal['ROC-AUC']*100:.2f}%",
            f"{metrics_bal['TN']}",
            f"{metrics_bal['FP']}",
            f"{metrics_bal['FN']}",
            f"{metrics_bal['TP']}"
        ]
    })

    print(comparison_df.to_string(index=False))

    print("\n" + "-" * 75)
    print("  Balanced Model Detailed Classification Report:")
    print("-" * 75)
    print(metrics_bal["report"])

    # ──────────────────────────────────────────────────────────────────────────
    # 4. SAVE MODEL
    # ──────────────────────────────────────────────────────────────────────────
    joblib.dump(rf_balanced, MODEL_PATH)
    print(f"\n[Saved] Final Balanced Random Forest Model -> {MODEL_PATH}")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. FEATURE IMPORTANCE (Balanced Model)
    # ──────────────────────────────────────────────────────────────────────────
    importances = pd.Series(rf_balanced.feature_importances_, index=FEATURES)
    importances_sorted = importances.sort_values(ascending=True)

    print("\n-- Feature Importance (Balanced Random Forest) ----------")
    for f, imp in importances_sorted.items():
        bar = "#" * int(imp * 40)
        print(f"  {f:<30} {imp*100:.2f}%  {bar}")

    # Chart 1: Feature Importance
    plt.figure(figsize=(8, 4.5))
    colors = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af"]
    importances_sorted.plot(kind="barh", color=colors, edgecolor="none")
    plt.title("Random Forest — Feature Importance (Balanced Dataset)", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Weight")
    for i, v in enumerate(importances_sorted):
        plt.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=10, fontweight="bold")
    plt.xlim(0, max(importances_sorted) * 1.18)
    plt.tight_layout()
    fi_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=130)
    plt.close()
    print(f"[Saved] Feature importance chart -> {fi_path}")

    # Chart 2: Confusion Matrix (Balanced Model)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        metrics_bal["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Burnout (0)", "Burnout (1)"],
        yticklabels=["No Burnout (0)", "Burnout (1)"],
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix — Balanced Model (N={len(X_test_bal)})", fontsize=12, fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=130)
    plt.close()
    print(f"[Saved] Confusion matrix chart -> {cm_path}")

    # Chart 3: Direct Side-by-Side Comparison Plot (Balance vs Imbalance)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot A: Metrics Bar Chart
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    imb_vals = [
        metrics_imb["Accuracy"] * 100,
        metrics_imb["Precision"] * 100,
        metrics_imb["Recall (Sensitivity)"] * 100,
        metrics_imb["F1-Score"] * 100,
        metrics_imb["ROC-AUC"] * 100
    ]
    bal_vals = [
        metrics_bal["Accuracy"] * 100,
        metrics_bal["Precision"] * 100,
        metrics_bal["Recall (Sensitivity)"] * 100,
        metrics_bal["F1-Score"] * 100,
        metrics_bal["ROC-AUC"] * 100
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, imb_vals, width, label="Imbalance (1000)", color="#94a3b8")
    rects2 = axes[0].bar(x + width/2, bal_vals, width, label="Balance (800)", color="#2563eb")

    axes[0].set_ylabel("Score (%)", fontweight="bold")
    axes[0].set_title("Performance Metrics Comparison", fontweight="bold", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names, rotation=15, ha="right", fontweight="bold")
    axes[0].set_ylim(85, 102)
    axes[0].legend(loc="lower right")

    # Add values on top of bars
    for r in rects1:
        h = r.get_height()
        axes[0].annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5)
    for r in rects2:
        h = r.get_height()
        axes[0].annotate(f"{h:.1f}%", (r.get_x() + r.get_width()/2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

    # Subplot B: Balanced Confusion Matrix
    sns.heatmap(
        metrics_bal["cm"],
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
        xticklabels=["No Burnout", "Burnout"],
        yticklabels=["No Burnout", "Burnout"],
        annot_kws={"size": 12, "weight": "bold"}
    )
    axes[1].set_title("Balanced Test Set Confusion Matrix", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("True Label", fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontweight="bold")

    fig.suptitle("Supervisor Review: Performance Comparison (Imbalance vs. Balance)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    comp_plot_path = os.path.join(MODEL_DIR, "comparison_balance_vs_imbalance.png")
    plt.savefig(comp_plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Comparison plot -> {comp_plot_path}")

    print("\nTraining and evaluation successfully completed! All models and charts saved. ✅")
    return rf_balanced


if __name__ == "__main__":
    main()
