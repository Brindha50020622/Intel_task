import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load the trained model and selected threshold
saved = joblib.load(BASE_DIR / "final_escalation_model.joblib")
model = saved["model"]
features = saved["features"]
threshold = saved["threshold"]

# Load prepared audio features
df = pd.read_csv(BASE_DIR / "dataset" / "prepared_features.csv")

# Prepare features
X = df[features].replace([np.inf, -np.inf], np.nan)
valid = X.notna().all(axis=1)

# Initialize output columns
df["escalation_probability"] = np.nan
df["predicted_escalation"] = 0

# Predict probabilities for valid rows
probabilities = model.predict_proba(X.loc[valid])[
    :, list(model.classes_).index(1)
]

df.loc[valid, "escalation_probability"] = probabilities
df.loc[valid, "predicted_escalation"] = (
    probabilities >= threshold
).astype(int)

# Save complete predictions
df.to_csv(BASE_DIR / "escalation_predictions.csv", index=False)

# Generate timestamped JSON output
output = []

for file_name, group in df.groupby("file_name"):
    segments = []

    for _, row in group.sort_values("start").iterrows():
        probability = row["escalation_probability"]

        if pd.isna(probability):
            continue

        segments.append({
            "start": float(row["start"]),
            "end": float(row["end"]),
            "escalation_probability": round(float(probability), 4),
            "predicted_escalation": int(row["predicted_escalation"])
        })

    output.append({
        "file_name": file_name,
        "segments": segments
    })

import json

with open(BASE_DIR / "escalation_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Model:", saved["model_name"])
print("Threshold:", threshold)
print("Total segments:", len(df))
print("Predicted escalations:", int(df["predicted_escalation"].sum()))
print("Saved escalation_predictions.csv")
print("Saved escalation_output.json")
