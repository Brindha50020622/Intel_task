import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "dataset" / "audio_features.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "prepared_features.csv"

df = pd.read_csv(INPUT_FILE)

# Remove rows without labels
df = df.dropna(subset=["escalation_label"])

# Select audio features only
feature_columns = [
    "duration",
    "rms_energy",
    "zero_crossing_rate",
    "pitch_mean",
    "spectral_centroid"
]

# Add the MFCC features
feature_columns += [f"mfcc_{i}" for i in range(1, 14)]

X = df[feature_columns]
y = df["escalation_label"].astype(int)

# Check for missing values
print("Missing feature values:", X.isnull().sum().sum())

# Save prepared dataset
prepared = X.copy()
prepared["escalation_label"] = y
prepared["file_name"] = df["file_name"]
prepared["start"] = df["start"]
prepared["end"] = df["end"]

prepared.to_csv(OUTPUT_FILE, index=False)

print("\nDataset preparation completed!")
print("Total segments:", len(prepared))
print("Number of audio features:", len(feature_columns))
print("\nLabel distribution:")
print(y.value_counts())
print("\nSaved to:", OUTPUT_FILE)
