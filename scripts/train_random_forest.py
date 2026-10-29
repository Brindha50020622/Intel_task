
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.base import clone
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "prepared_features.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Audio features only
feature_columns = [
    "duration",
    "rms_energy",
    "zero_crossing_rate",
    "pitch_mean",
    "spectral_centroid"
]
feature_columns += [f"mfcc_{i}" for i in range(1, 14)]

X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
y = df["escalation_label"]
groups = df["file_name"]

# Remove invalid feature rows
valid = X.notna().all(axis=1) & y.notna() & groups.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(int).reset_index(drop=True)
groups = groups.loc[valid].reset_index(drop=True)

print("Total segments:", len(X))
print("Unique calls:", groups.nunique())
print("\nLabel distribution:")
print(y.value_counts())

# Check whether five folds are possible
if groups.nunique() < 5:
    raise ValueError("At least 5 different calls are required.")

# Grouped 5-fold cross-validation
cv = GroupKFold(n_splits=5)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2
)

scores = []
all_true = []
all_pred = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Check that both classes are present in training
    if y_train.nunique() < 2:
        print(f"Fold {fold}: skipped because training has only one class.")
        continue

    fold_model = clone(model)
    fold_model.fit(X_train, y_train)

    predictions = fold_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    scores.append([accuracy, precision, recall, f1])
    all_true.extend(y_test.tolist())
    all_pred.extend(predictions.tolist())

    print(f"\nFold {fold}")
    print("Training calls:", groups.iloc[train_idx].nunique())
    print("Testing calls:", groups.iloc[test_idx].nunique())
    print("Test labels:", sorted(y_test.unique().tolist()))
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

if not scores:
    raise ValueError(
        "No valid folds completed. Check the label distribution across calls."
    )

scores = np.array(scores)

print("\n===== CROSS-VALIDATION RESULTS =====")
print(f"Completed folds: {len(scores)} / 5")
print(f"Mean Accuracy:  {scores[:, 0].mean():.3f}")
print(f"Mean Precision: {scores[:, 1].mean():.3f}")
print(f"Mean Recall:    {scores[:, 2].mean():.3f}")
print(f"Mean F1-score:  {scores[:, 3].mean():.3f}")

print("\nCombined out-of-fold classification report:")
print(classification_report(
    all_true,
    all_pred,
    labels=[0, 1],
    target_names=["Not escalated", "Escalated"],
    zero_division=0
))

# Train final baseline model on all available labelled segments
final_model = clone(model)
final_model.fit(X, y)

model_path = BASE_DIR / "random_forest_model.joblib"
joblib.dump(
    {
        "model": final_model,
        "feature_columns": feature_columns
    },
    model_path
)

print("\nFinal baseline model saved to:", model_path)
