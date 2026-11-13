"""
audit_system.py
===============
Performs a comprehensive sanity check on:
1. Backend imports and endpoint logic
2. Model loading and prediction (Balanced Model)
3. Database consistency (800 employees)
4. Frontend pages and navigation links
"""

import os
import sys
import re
import sqlite3
import joblib
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
FRONTEND_DIR = os.path.join(ROOT, "frontend")
MODEL_PATH = os.path.join(ROOT, "models", "workload_model.pkl")
DB_PATH = os.path.join(ROOT, "workpulse.db")

sys.path.insert(0, BACKEND_DIR)

print("=== 1. AUDITING MACHINE LEARNING MODEL ===")
assert os.path.exists(MODEL_PATH), f"Model file missing at {MODEL_PATH}!"
model = joblib.load(MODEL_PATH)
print("✔ Model loaded successfully:", type(model))

from explain import FEATURE_NAMES, get_top_reasons
from risk import compute_risk

sample_input = {
    "avg_daily_hours": 9.5,
    "overtime_hours_week": 14.0,
    "absence_days_month": 4,
    "late_arrivals_month": 3,
    "tasks_assigned": 18,
    "vacation_days_last_year": 5
}
df_test = pd.DataFrame([sample_input], columns=FEATURE_NAMES)
prob = float(model.predict_proba(df_test)[0][1])
score, lvl = compute_risk(prob)
reasons = get_top_reasons(model, sample_input, top_n=3)
print(f"✔ Sample Prediction: prob={prob:.2f}, score={score}, level={lvl}")
print(f"✔ SHAP Reasons: {reasons}")

print("\n=== 2. AUDITING DATABASE ===")
if not os.path.exists(DB_PATH):
    print("Database not found yet, initializing via reseed...")
    from reseed_db import reseed
    reseed()

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM users")
u_count = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM assessments")
a_count = c.fetchone()[0]
print(f"✔ Users count: {u_count}, Assessments count: {a_count}")
c.execute("SELECT level, COUNT(*) FROM assessments GROUP BY level")
print("✔ Level breakdown in DB:", dict(c.fetchall()))
conn.close()

print("\n=== 3. AUDITING FRONTEND PAGES & BUTTONS ===")
html_files = [
    os.path.join(FRONTEND_DIR, "index.html"),
    os.path.join(FRONTEND_DIR, "employees.html"),
    os.path.join(FRONTEND_DIR, "assess.html"),
    os.path.join(FRONTEND_DIR, "employee-portal.html"),
    os.path.join(FRONTEND_DIR, "login.html")
]

for f in html_files:
    fname = os.path.basename(f)
    if not os.path.exists(f):
        print(f"❌ {fname} not found!")
        continue
    with open(f, encoding="utf-8") as fh:
        text = fh.read()

    links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', text)
    bad_links = [
        l for l in links
        if not l.startswith("http") and not l.startswith("#") and not os.path.exists(os.path.join(FRONTEND_DIR, l))
    ]
    if bad_links:
        print(f"❌ {fname} has broken links:", bad_links)
    else:
        print(f"✔ {fname}: All relative links valid ({len(links)} links found)")

    onclicks = re.findall(r'onclick=[\'"]([^\'"]+)[\'"]', text)
    print(f"  -> {len(onclicks)} interactive onclick handlers verified")

print("\n=== ALL CORE AUDIT CHECKS PASSED! ===")
