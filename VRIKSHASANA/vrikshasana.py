"""
Vrikshasana Data Cleaning Fix
-------------------------------
Problem: Official threshold table specifies shoulder angle must be either
3-50 deg (arms down/at sides) OR 150-180 deg (arms overhead) - the 51-149 deg
range (horizontal/T-pose arms) should be Incorrect. But the trained model
almost entirely ignores shoulder angle (importance ~5.8% combined) because
the Incorrect-class training data has only 2 examples in that gap - nowhere
near enough to teach the model to reject it. Confirmed via live testing:
horizontal arms + proper knee/hip lift still shows "Correct".

Fix: Add synthetic negative examples explicitly covering the 51-149 deg
shoulder gap, paired with realistic knee/hip values (so the model learns
"good leg lift + bad arm position = still Incorrect", not just "bad arm
position in isolation").
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

POSE_NAME = "Vrikshasana"
MODEL_FILENAME = "vrikshasana_rf.pkl"

FEATURES = ["left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
TARGET_COL = "target"
RANDOM_STATE = 42

RAW_CORRECT_FILE = "../CSV/correct_vrikshasana (1).csv"
RAW_INCORRECT_FILE = "../CSV/incorrect_vrikshasana (1).csv"

# Official threshold table values
SHOULDER_GAP_MIN = 51
SHOULDER_GAP_MAX = 149


def is_shoulder_gap_violation(df):
    """Rows where EITHER shoulder falls in the excluded 51-149 deg zone."""
    return (
        (df["left_shoulder"].between(SHOULDER_GAP_MIN, SHOULDER_GAP_MAX)) |
        (df["right_shoulder"].between(SHOULDER_GAP_MIN, SHOULDER_GAP_MAX))
    )


raw_correct = pd.read_csv(RAW_CORRECT_FILE)
raw_incorrect = pd.read_csv(RAW_INCORRECT_FILE)

print(f"Before cleaning -> Correct: {raw_correct.shape} | Incorrect: {raw_incorrect.shape}")

# Remove the handful of contaminated "Correct" rows that already violate the gap rule
gap_mask = is_shoulder_gap_violation(raw_correct)
removed_count = gap_mask.sum()
raw_correct_clean = raw_correct[~gap_mask].reset_index(drop=True)

print(f"Removed {removed_count} shoulder-gap-violating rows from 'Correct' class "
      f"({removed_count/len(raw_correct)*100:.1f}%)")

raw_correct_clean = raw_correct_clean.drop_duplicates(subset=FEATURES).reset_index(drop=True)
raw_incorrect_clean = raw_incorrect.drop_duplicates(subset=FEATURES).reset_index(drop=True)
print(f"After dedup -> Correct: {raw_correct_clean.shape} | Incorrect: {raw_incorrect_clean.shape}")

# ---------------------------------------------------------------
# Synthetic negatives: realistic knee/hip lift (sampled from real Correct
# data) PAIRED with shoulder angles drawn from the excluded gap. This
# teaches "even with a good leg lift, bad arm position = Incorrect".
# ---------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
N_SYNTHETIC = 400

# Sample realistic knee/hip combos from the real (cleaned) correct data
sampled_base = raw_correct_clean.sample(n=N_SYNTHETIC, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)

synthetic_gap = pd.DataFrame({
    "left_knee": sampled_base["left_knee"],
    "right_knee": sampled_base["right_knee"],
    "left_hip": sampled_base["left_hip"],
    "right_hip": sampled_base["right_hip"],
    # Shoulder angles drawn from the excluded gap for both arms
    "left_shoulder": rng.uniform(SHOULDER_GAP_MIN, SHOULDER_GAP_MAX, N_SYNTHETIC),
    "right_shoulder": rng.uniform(SHOULDER_GAP_MIN, SHOULDER_GAP_MAX, N_SYNTHETIC),
})
synthetic_gap[TARGET_COL] = 0  # Incorrect

print(f"\nAdded {N_SYNTHETIC} synthetic gap-violation examples (realistic knee/hip + bad shoulder) "
      f"to Incorrect class.")

raw_incorrect_clean = pd.concat([raw_incorrect_clean, synthetic_gap], ignore_index=True)
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
print("Model trained on cleaned + gap-negative data.")

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
# THE REAL TEST: reproduce the exact failing screenshots (Images 16, 18)
# ---------------------------------------------------------------
test_cases = pd.DataFrame([
    {"name": "Image 16: leg lift + arms horizontal (74/81 deg)",
     "left_knee": 46.5, "right_knee": 171.4, "left_hip": 115.4, "right_hip": 174.1,
     "left_shoulder": 74.4, "right_shoulder": 80.8},
    {"name": "Image 18: leg lift + arms horizontal (78/80 deg)",
     "left_knee": 174.6, "right_knee": 58.8, "left_hip": 174.4, "right_hip": 105.4,
     "left_shoulder": 78.5, "right_shoulder": 79.9},
    {"name": "Image 17: leg lift + arms overhead (correct, should stay Correct)",
     "left_knee": 45.5, "right_knee": 174.9, "left_hip": 120.2, "right_hip": 178.8,
     "left_shoulder": 179.7, "right_shoulder": 175.2},
])

print("\n=== Reproducing exact screenshots ===")
for _, row in test_cases.iterrows():
    X_case = row[FEATURES].to_frame().T
    pred = rf.predict(X_case)[0]
    proba = rf.predict_proba(X_case)[0]
    label = "Correct" if pred == 1 else "Incorrect"
    print(f"  {row['name']}: {label} (P(Incorrect)={proba[0]*100:.1f}%, P(Correct)={proba[1]*100:.1f}%)")

# ---------------------------------------------------------------
# Confusion matrix, feature importance, confidence
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {POSE_NAME} (Cleaned)")
plt.savefig(f"confusion_matrix_{POSE_NAME.lower()}_cleaned.png", dpi=150, bbox_inches="tight")
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
    "note": (f"Removed {removed_count} rows violating the official shoulder threshold "
             f"(must be 3-50 or 150-180 deg) from Correct class. Added {N_SYNTHETIC} synthetic "
             f"negative examples pairing realistic knee/hip values with shoulder angles in the "
             f"excluded 51-149 deg gap, so arm position is now properly weighted in the decision.")
}

metadata_filename = MODEL_FILENAME.replace(".pkl", "_metadata.json")
with open(metadata_filename, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nModel saved as {MODEL_FILENAME}")
print(f"Metadata saved as {metadata_filename}")
