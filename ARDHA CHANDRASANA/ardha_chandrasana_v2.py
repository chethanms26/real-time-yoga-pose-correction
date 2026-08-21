"""
Ardha Chandrasana Data Cleaning Fix v2
-----------------------------------------
v1 fix only removed standing-contaminated rows with arms DOWN (shoulder<40),
which missed rows where the person stood straight but had arms RAISED
(shoulder>100) - still not a real Half Moon pose (no hip bend, no leg lift),
but slipped through because the filter combined hip/knee straightness with
a low shoulder requirement.

Root insight: genuine Ardha Chandrasana ALWAYS requires a bent hip (leaning +
one leg lifted). Straight hips (>160 deg) should NEVER be Correct, regardless
of arm position. This version filters and synthesizes negatives based on
hip/knee straightness ALONE, with a wider margin (160 instead of 165) so
near-boundary real-world readings don't slip through either.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

POSE_NAME = "Ardha Chandrasana"
MODEL_FILENAME = "ardha_chandrasana_rf.pkl"

FEATURES = ["left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
TARGET_COL = "target"
RANDOM_STATE = 42

RAW_CORRECT_FILE = "../CSV/correct_ardha_chandrasana (1).csv"
RAW_INCORRECT_FILE = "../CSV/incorrect_ardha_chandrasana (1).csv"

# Widened threshold - based on hip/knee straightness ONLY, regardless of arm position
STRAIGHT_KNEE_HIP_MIN = 160  # was 165 - wider margin so boundary noise doesn't slip through


def is_straight_pose(df):
    """Rows where hips AND knees are straight - not a real Half Moon regardless of arms."""
    return (
        (df["left_knee"] > STRAIGHT_KNEE_HIP_MIN) &
        (df["right_knee"] > STRAIGHT_KNEE_HIP_MIN) &
        (df["left_hip"] > STRAIGHT_KNEE_HIP_MIN) &
        (df["right_hip"] > STRAIGHT_KNEE_HIP_MIN)
    )


raw_correct = pd.read_csv(RAW_CORRECT_FILE)
raw_incorrect = pd.read_csv(RAW_INCORRECT_FILE)

print(f"Before cleaning -> Correct: {raw_correct.shape} | Incorrect: {raw_incorrect.shape}")

straight_mask = is_straight_pose(raw_correct)
removed_count = straight_mask.sum()
raw_correct_clean = raw_correct[~straight_mask].reset_index(drop=True)

print(f"Removed {removed_count} straight-hip/knee rows from 'Correct' class "
      f"({removed_count/len(raw_correct)*100:.1f}% of original Correct data)")
print(f"After cleaning  -> Correct: {raw_correct_clean.shape} | Incorrect: {raw_incorrect.shape}")

raw_correct_clean = raw_correct_clean.drop_duplicates(subset=FEATURES).reset_index(drop=True)
raw_incorrect_clean = raw_incorrect.drop_duplicates(subset=FEATURES).reset_index(drop=True)

print(f"After dedup     -> Correct: {raw_correct_clean.shape} | Incorrect: {raw_incorrect_clean.shape}")

# ---------------------------------------------------------------
# Synthetic negatives: straight hip/knee, ANY shoulder angle (0-180)
# This covers BOTH arms-down AND arms-up standing variants.
# ---------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
N_SYNTHETIC = 600  # more than v1, covering a wider shoulder range

synthetic_straight = pd.DataFrame({
    "left_knee":  rng.normal(177, 3, N_SYNTHETIC).clip(160, 180),
    "right_knee": rng.normal(177, 3, N_SYNTHETIC).clip(160, 180),
    "left_hip":   rng.normal(177, 3, N_SYNTHETIC).clip(160, 180),
    "right_hip":  rng.normal(177, 3, N_SYNTHETIC).clip(160, 180),
    # Shoulder spans the FULL range - arms down, up, or anywhere between -
    # all Incorrect if hips/knees are straight, since real pose needs hip bend
    "left_shoulder":  rng.uniform(0, 180, N_SYNTHETIC),
    "right_shoulder": rng.uniform(0, 180, N_SYNTHETIC),
})
synthetic_straight[TARGET_COL] = 0  # Incorrect

print(f"\nAdded {N_SYNTHETIC} synthetic straight-hip/knee examples (full shoulder range) "
      f"to Incorrect class.")

raw_incorrect_clean = pd.concat([raw_incorrect_clean, synthetic_straight], ignore_index=True)
print(f"Incorrect class after synthetic addition: {raw_incorrect_clean.shape}")

# ---------------------------------------------------------------
# Rebuild split
# ---------------------------------------------------------------
def split_70_15_15(df, random_state=RANDOM_STATE):
    train, temp = train_test_split(df, test_size=0.30, random_state=random_state)
    val, test = train_test_split(temp, test_size=0.50, random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

correct_train, correct_val, correct_test = split_70_15_15(raw_correct_clean)
incorrect_train, incorrect_val, incorrect_test = split_70_15_15(raw_incorrect_clean)

for split in ["train", "val", "test"]:
    os.makedirs(split, exist_ok=True)

correct_train.to_csv("train/correct_train.csv", index=False)
incorrect_train.to_csv("train/incorrect_train.csv", index=False)
correct_val.to_csv("val/correct_val.csv", index=False)
incorrect_val.to_csv("val/incorrect_val.csv", index=False)
correct_test.to_csv("test/correct_test.csv", index=False)
incorrect_test.to_csv("test/incorrect_test.csv", index=False)

print("\nNew split sizes:")
print("Train:", len(correct_train) + len(incorrect_train))
print("Val  :", len(correct_val) + len(incorrect_val))
print("Test :", len(correct_test) + len(incorrect_test))

# ---------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------
def load_merge(split_name):
    c = pd.read_csv(f"{split_name}/correct_{split_name}.csv")
    i = pd.read_csv(f"{split_name}/incorrect_{split_name}.csv")
    return pd.concat([c, i], ignore_index=True)

train_df = load_merge("train")
val_df = load_merge("val")
test_df = load_merge("test")

train_set = set(map(tuple, train_df[FEATURES].round(4).values))
val_set   = set(map(tuple, val_df[FEATURES].round(4).values))
test_set  = set(map(tuple, test_df[FEATURES].round(4).values))

overlap_train_val  = len(train_set & val_set)
overlap_train_test = len(train_set & test_set)
overlap_val_test   = len(val_set & test_set)

print("\nLeakage check:")
print("Train ∩ Val overlap :", overlap_train_val)
print("Train ∩ Test overlap:", overlap_train_test)
print("Val ∩ Test overlap  :", overlap_val_test)

assert overlap_train_val == 0
assert overlap_train_test == 0
assert overlap_val_test == 0
print("No leakage. Safe to train.\n")

# ---------------------------------------------------------------
# Train
# ---------------------------------------------------------------
X_train, y_train = train_df[FEATURES], train_df[TARGET_COL]
X_val,   y_val   = val_df[FEATURES],   val_df[TARGET_COL]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET_COL]

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced")
rf.fit(X_train, y_train)
print("Model trained on cleaned + widened-negative data.")

# ---------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------
train_pred = rf.predict(X_train)
val_pred   = rf.predict(X_val)
test_pred  = rf.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy   = accuracy_score(y_val, val_pred)
test_accuracy  = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall    = recall_score(y_test, test_pred)
test_f1        = f1_score(y_test, test_pred)

print(f"\nTrain Accuracy      : {train_accuracy*100:.2f}%")
print(f"Validation Accuracy : {val_accuracy*100:.2f}%")
print(f"Test Accuracy       : {test_accuracy*100:.2f}%")
print(f"Test Precision      : {test_precision*100:.2f}%")
print(f"Test Recall         : {test_recall*100:.2f}%")
print(f"Test F1             : {test_f1*100:.2f}%")
print("\nFull Classification Report (Test)")
print(classification_report(y_test, test_pred, target_names=["Incorrect", "Correct"]))

# ---------------------------------------------------------------
# THE REAL TEST: reproduce the exact failing screenshots
# ---------------------------------------------------------------
test_cases = pd.DataFrame([
    {"name": "Standing, arms down (Image 2)",
     "left_knee": 176.8, "right_knee": 178.4, "left_hip": 177.5, "right_hip": 173.3,
     "left_shoulder": 17.2, "right_shoulder": 14.6},
    {"name": "Standing, arms raised overhead (Image 3)",
     "left_knee": 173.9, "right_knee": 175.8, "left_hip": 177.7, "right_hip": 173.3,
     "left_shoulder": 161.2, "right_shoulder": 160.9},
    {"name": "Standing, arms up circle (Image 4)",
     "left_knee": 179.1, "right_knee": 175.8, "left_hip": 178.8, "right_hip": 175.3,
     "left_shoulder": 180.0, "right_shoulder": 166.2},
    {"name": "Near-boundary standing (Image 6)",
     "left_knee": 173.9, "right_knee": 176.0, "left_hip": 171.4, "right_hip": 163.8,
     "left_shoulder": 12.1, "right_shoulder": 16.4},
])

print("\n=== Reproducing exact failing screenshots ===")
for _, row in test_cases.iterrows():
    X_case = row[FEATURES].to_frame().T
    pred = rf.predict(X_case)[0]
    proba = rf.predict_proba(X_case)[0]
    label = "Correct" if pred == 1 else "Incorrect"
    status = "FIXED" if pred == 0 else "STILL BROKEN"
    print(f"  {row['name']}: {label} (P(Incorrect)={proba[0]*100:.1f}%) -> {status}")

# ---------------------------------------------------------------
# Confusion matrix, feature importance, confidence
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {POSE_NAME} (Cleaned v2)")
plt.savefig(f"confusion_matrix_{POSE_NAME.lower().replace(' ', '_')}_cleaned_v2.png", dpi=150, bbox_inches="tight")
plt.close()

importance_df = pd.DataFrame({
    "Feature": FEATURES, "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)
print("\nFeature Importance:")
print(importance_df)

test_probs = rf.predict_proba(X_test)
test_confidence = test_probs.max(axis=1)
print(f"\nAverage confidence : {test_confidence.mean()*100:.2f}%")
print(f"Minimum confidence : {test_confidence.min()*100:.2f}%")

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
joblib.dump(rf, MODEL_FILENAME)

metadata = {
    "pose_name": POSE_NAME,
    "features": FEATURES,
    "target_mapping": {"0": "Incorrect", "1": "Correct"},
    "train_accuracy": round(train_accuracy, 4),
    "val_accuracy": round(val_accuracy, 4),
    "test_accuracy": round(test_accuracy, 4),
    "test_precision": round(test_precision, 4),
    "test_recall": round(test_recall, 4),
    "test_f1_score": round(test_f1, 4),
    "n_estimators": rf.n_estimators,
    "random_state": rf.random_state,
    "data_cleaned": True,
    "fix_version": 2,
    "note": (f"v2: Removed {removed_count} straight-hip/knee rows ({removed_count/len(raw_correct)*100:.1f}% "
             f"of original Correct class) mislabeled as Correct, regardless of arm position. Added {N_SYNTHETIC} "
             f"synthetic straight-pose examples spanning the FULL shoulder range (0-180 deg) to Incorrect class. "
             f"Fixes v1 gap where standing-with-arms-raised was still misclassified as Correct.")
}

metadata_filename = MODEL_FILENAME.replace(".pkl", "_metadata.json")
with open(metadata_filename, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nModel saved as {MODEL_FILENAME}")
print(f"Metadata saved as {metadata_filename}")
