"""
balance_dataset.py
==================
Implements supervisor Dr. Amir Al-Attar's directive:
1. Balance dataset by undersampling majority class (No Burnout 0) from 1,000 to 800 records (400 class 0, 400 class 1).
2. Maintain strict train/test split integrity:
   - 80% Train = 640 records (320 class 0, 320 class 1)
   - 20% Test  = 160 records (80 class 0, 80 class 1)
3. Ensure the 160 test records are strictly drawn from the original 200 test set (zero data leakage).
4. Save:
   - data/employees_1000_imbalanced.csv  (Backup of original 1000-record dataset)
   - data/employees.csv                  (Full 800 balanced records)
   - data/train.csv                      (640 training records)
   - data/test.csv                       (160 testing records)
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ORIGINAL_CSV = os.path.join(DATA_DIR, "employees.csv")
BACKUP_CSV = os.path.join(DATA_DIR, "employees_1000_imbalanced.csv")
BALANCED_CSV = os.path.join(DATA_DIR, "employees.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

SEED = 42

def balance_and_split():
    print("=" * 70)
    print("  TAWAZON — DATA BALANCING & SPLITTING PIPELINE")
    print("  Directive: Dr. Amir Al-Attar (Supervisor Follow-up)")
    print("=" * 70)

    # 1. Load original data
    if not os.path.exists(BACKUP_CSV):
        if os.path.exists(ORIGINAL_CSV):
            print(f"[Backup] Creating backup of original 1000 dataset -> {BACKUP_CSV}")
            shutil.copyfile(ORIGINAL_CSV, BACKUP_CSV)
        else:
            raise FileNotFoundError(f"Original dataset not found at {ORIGINAL_CSV}")
    
    df_orig = pd.read_csv(BACKUP_CSV)
    print(f"\n[Original Dataset] Loaded: {len(df_orig)} records")
    print(f"  Class 0 (No Burnout): {(df_orig['burnout'] == 0).sum()} ({(df_orig['burnout'] == 0).mean()*100:.1f}%)")
    print(f"  Class 1 (Burnout):    {(df_orig['burnout'] == 1).sum()} ({(df_orig['burnout'] == 1).mean()*100:.1f}%)")

    # 2. Re-create the exact original 80/20 train/test split (800 train, 200 test)
    train_orig, test_orig = train_test_split(
        df_orig,
        test_size=0.20,
        random_state=SEED,
        stratify=df_orig['burnout']
    )
    print(f"\n[Original Split (80/20)]")
    print(f"  Original Train: {len(train_orig)} -> {train_orig['burnout'].value_counts().to_dict()}")
    print(f"  Original Test:  {len(test_orig)} -> {test_orig['burnout'].value_counts().to_dict()}")

    # 3. Sample exactly 80 from class 0 and 80 from class 1 STRICTLY from original test set
    test_0 = test_orig[test_orig['burnout'] == 0].sample(n=80, random_state=SEED)
    test_1 = test_orig[test_orig['burnout'] == 1].sample(n=80, random_state=SEED)
    test_balanced = pd.concat([test_0, test_1]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 4. Sample exactly 320 from class 0 and 320 from class 1 STRICTLY from original train set
    train_0 = train_orig[train_orig['burnout'] == 0].sample(n=320, random_state=SEED)
    train_1 = train_orig[train_orig['burnout'] == 1].sample(n=320, random_state=SEED)
    train_balanced = pd.concat([train_0, train_1]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # 5. Full balanced dataset (800 records: 400 class 0, 400 class 1)
    full_balanced = pd.concat([train_balanced, test_balanced]).reset_index(drop=True)

    # 6. Verification and Assertions
    print(f"\n[Balanced Dataset Verification]")
    print(f"  Full Balanced: {len(full_balanced)} records -> {full_balanced['burnout'].value_counts().to_dict()}")
    print(f"  Train (80%):   {len(train_balanced)} records -> {train_balanced['burnout'].value_counts().to_dict()}")
    print(f"  Test (20%):    {len(test_balanced)} records -> {test_balanced['burnout'].value_counts().to_dict()}")

    assert len(full_balanced) == 800, f"Expected 800 records, got {len(full_balanced)}"
    assert len(train_balanced) == 640, f"Expected 640 train records, got {len(train_balanced)}"
    assert len(test_balanced) == 160, f"Expected 160 test records, got {len(test_balanced)}"
    assert (full_balanced['burnout'] == 0).sum() == 400, "Class 0 count is not 400"
    assert (full_balanced['burnout'] == 1).sum() == 400, "Class 1 count is not 400"

    # Leakage checks
    test_ids_orig = set(test_orig['employee_id'])
    test_ids_bal = set(test_balanced['employee_id'])
    assert test_ids_bal.issubset(test_ids_orig), "CRITICAL ERROR: Test records leak outside original test set!"

    train_ids_orig = set(train_orig['employee_id'])
    train_ids_bal = set(train_balanced['employee_id'])
    assert train_ids_bal.issubset(train_ids_orig), "CRITICAL ERROR: Train records leak outside original train set!"
    assert test_ids_bal.isdisjoint(train_ids_bal), "CRITICAL ERROR: Overlap between train and test sets!"

    print("  [Pass] Zero data leakage verified.")
    print("  [Pass] Test set (160) is a strict 100% subset of original test set (200).")
    print("  [Pass] Exact 50/50 balance in both Train and Test splits.")

    # 7. Save files
    full_balanced.to_csv(BALANCED_CSV, index=False)
    print(f"\n[Saved] Full 800 balanced employees -> {BALANCED_CSV}")

    train_balanced.to_csv(TRAIN_CSV, index=False)
    print(f"[Saved] Train dataset (640 records) -> {TRAIN_CSV}")

    test_balanced.to_csv(TEST_CSV, index=False)
    print(f"[Saved] Test dataset (160 records)  -> {TEST_CSV}")

    print("\nData balancing and file generation completed successfully! ✅")

if __name__ == "__main__":
    balance_and_split()
