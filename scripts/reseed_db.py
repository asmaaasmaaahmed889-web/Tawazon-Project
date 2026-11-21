"""
reseed_db.py
============
High-Speed Database Seeder for Tawazon (160 Test Employees Active Workforce)
Reseeds the database with clean test employee records (160 records) and their
assessments predicted by the updated Random Forest model.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import joblib
from database import SessionLocal, User, Assessment, hash_password, init_db
from risk import compute_risk
from explain import get_top_reasons

DATA_PATH = os.path.join(ROOT, "data", "test.csv")
MODEL_PATH = os.path.join(ROOT, "models", "workload_model.pkl")

NUMERIC_FEATURES = [
    "avg_daily_hours",
    "overtime_hours_week",
    "absence_days_month",
    "late_arrivals_month",
    "tasks_assigned",
    "vacation_days_last_year",
]


def reseed():
    print("=" * 65)
    print("  TAWAZON: High-Speed Database Reseeder (160 Test Employees)")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"[Error] {DATA_PATH} not found.")
        return

    init_db()
    db = SessionLocal()

    try:
        # 1. Load ML Model
        print("[1/4] Loading Balanced ML Model...")
        workload_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        if workload_model is None:
            print("[Warning] workload_model.pkl not found! Using fallback.")

        # 2. Reset employee data (preserve admin)
        print("[2/4] Resetting employee and assessment tables...")
        db.query(Assessment).delete()
        db.query(User).filter(User.role == "employee").delete()
        db.commit()

        # 3. Read dataset
        df = pd.read_csv(DATA_PATH)
        print(f"[3/4] Loaded {len(df)} test employee records from {DATA_PATH}")

        # Batch ML predictions
        probs = workload_model.predict_proba(df[NUMERIC_FEATURES])[:, 1] if workload_model else [0.5] * len(df)
        default_pwd_hash = hash_password("123456")

        # 4. Insert records
        print(f"[4/4] Inserting {len(df)} users and corresponding assessments...")
        users_to_add = []
        assessments_to_add = []

        for idx, row in df.iterrows():
            emp_id = str(row["employee_id"])
            full_name = str(row["name"])
            username = emp_id.lower()
            gender_val = str(row["gender"]) if "gender" in row and pd.notna(row["gender"]) else None
            age_val = int(row["age"]) if "age" in row and pd.notna(row["age"]) else None

            user = User(
                username=username,
                full_name=full_name,
                gender=gender_val,
                age=age_val,
                role="employee",
                hashed_password=default_pwd_hash
            )
            db.add(user)
            db.flush()

            w_prob = float(probs[idx])
            score, level = compute_risk(w_prob)
            reasons = get_top_reasons(workload_model, row.to_dict(), top_n=3)

            assessment = Assessment(
                user_id=user.id,
                avg_daily_hours=float(row["avg_daily_hours"]),
                overtime_hours_week=float(row["overtime_hours_week"]),
                absence_days_month=int(row["absence_days_month"]),
                late_arrivals_month=int(row["late_arrivals_month"]),
                tasks_assigned=int(row["tasks_assigned"]),
                vacation_days_last_year=int(row["vacation_days_last_year"]),
                feedback=str(row.get("feedback", "")),
                risk_score=score,
                level=level,
                workload_probability=round(w_prob, 4),
                sentiment_risk=0.0,
                top_reasons=json.dumps(reasons),
            )
            db.add(assessment)

        db.commit()
        print(f"\n[Success] Database reseeded with {len(df)} test employees! ✅")

        # Quick breakdown
        counts = {"Low": 0, "Medium": 0, "High": 0}
        for a in db.query(Assessment).all():
            counts[a.level] = counts.get(a.level, 0) + 1
        print(f"  Risk distribution in DB: {counts}")

    except Exception as e:
        db.rollback()
        print(f"[Error] Reseeding failed: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    reseed()
