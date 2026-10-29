import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 1. Setup
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "prepared_features.csv"

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")
print("Using device:", device)

df = pd.read_csv(DATA_FILE)

features = [
    "duration", "rms_energy", "zero_crossing_rate",
    "pitch_mean", "spectral_centroid"
] + [f"mfcc_{i}" for i in range(1, 14)]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=features + [
    "escalation_label", "file_name", "start", "end"
]).copy()

df["escalation_label"] = df["escalation_label"].astype(int)

# Each sequence contains 10 consecutive segments from one call.
SEQ_LEN = 10
EPOCHS = 12
BATCH_SIZE = 64

print("Segments:", len(df))
print("Calls:", df["file_name"].nunique())
print("Positive labels:", int(df["escalation_label"].sum()))


# -----------------------------
# 2. Create sequences per call
# -----------------------------
def create_sequences(frame, scaler=None, fit_scaler=False):
    sequences = []
    labels = []
    call_ids = []

    if fit_scaler:
        scaler.fit(frame[features])

    for call_id, group in frame.groupby("file_name",sort=True):
        group = group.sort_values("start").reset_index(drop=True)

        values = group[features].to_numpy(dtype=np.float32)
        values = scaler.transform(values).astype(np.float32)
        targets = group["escalation_label"].to_numpy(dtype=np.float32)

        for start_idx in range(len(group)):
            end_idx = min(start_idx + SEQ_LEN, len(group))

            seq = values[start_idx:end_idx]
            target = targets[start_idx]

            # Pad short sequences with zeros.
            if len(seq) < SEQ_LEN:
                padding = np.zeros(
                    (SEQ_LEN - len(seq), len(features)),
                    dtype=np.float32
                )
                seq = np.vstack([seq, padding])

            sequences.append(seq)
            labels.append(target)
            call_ids.append(call_id)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(call_ids)
    )


# -----------------------------
# 3. Define LSTM model
# -----------------------------
class EscalationLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.lstm(x)

        # First output corresponds to the target segment.
        output = output[:, 0, :]
        output = self.dropout(output)

        return self.fc(output).squeeze(1)


# -----------------------------
# 4. Five-fold grouped CV
# -----------------------------
groups = df["file_name"].to_numpy()
cv = GroupKFold(n_splits=5)

oof_probabilities = np.full(len(df), np.nan)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(df, df["escalation_label"], groups=groups), start=1
):
    print(f"\nFold {fold}/5")

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Fit scaler only on training calls.
    scaler = StandardScaler()
    scaler.fit(train_df[features])

    X_train, y_train, train_calls = create_sequences(
        train_df, scaler=scaler
    )
    X_test, y_test, test_calls = create_sequences(
        test_df, scaler=scaler
    )

    # Map generated sequences back to their original rows.
    test_row_indices = []
    for call_id, group in test_df.groupby("file_name", sort=False):
        ordered_indices = group.sort_values("start").index.tolist()
        test_row_indices.extend(ordered_indices)

    # Sequence generation uses the same per-call sorted order.
    train_tensor = torch.tensor(X_train)
    label_tensor = torch.tensor(y_train)

    dataset = TensorDataset(train_tensor, label_tensor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = EscalationLSTM(len(features)).to(device)

    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(len(y_train) - positive_count, 1)

    pos_weight = torch.tensor(
        [min(negative_count / positive_count, 20.0)],
        dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test))
        test_probs = torch.sigmoid(test_logits).numpy()

    # Restore the probabilities to the original row positions.
    oof_probabilities[test_row_indices] = test_probs

    print("Fold training completed.")


# -----------------------------
# 5. Evaluate cross-validation
# -----------------------------
valid = ~np.isnan(oof_probabilities)

y_true = df["escalation_label"].to_numpy()[valid]
probs = oof_probabilities[valid]

print("\n========== LSTM RESULTS ==========")
print("Average precision:", round(
    average_precision_score(y_true, probs), 4
))

# Use a fixed threshold for the initial evaluation.
threshold = 0.5
predictions = (probs >= threshold).astype(int)

print("Threshold:", threshold)
print("Precision:", round(
    precision_score(y_true, predictions, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_true, predictions, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_true, predictions, zero_division=0), 4
))

# Save out-of-fold predictions for analysis.
result_df = df.copy()
result_df["lstm_probability"] = oof_probabilities
result_df.to_csv(BASE_DIR / "lstm_cv_predictions.csv", index=False)

print("Saved lstm_cv_predictions.csv")


# -----------------------------
# 6. Train final model on all calls
# -----------------------------
scaler = StandardScaler()
scaler.fit(df[features])

X_all, y_all, _ = create_sequences(df, scaler=scaler)

final_model = EscalationLSTM(len(features))
final_model.train()

positive_count = max(int(y_all.sum()), 1)
negative_count = max(len(y_all) - positive_count, 1)

pos_weight = torch.tensor(
    [min(negative_count / positive_count, 20.0)],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)

dataset = TensorDataset(
    torch.tensor(X_all),
    torch.tensor(y_all)
)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for epoch in range(EPOCHS):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(final_model(xb), yb)
        loss.backward()
        optimizer.step()

# Save model weights and scaler.
torch.save(
    {
        "state_dict": final_model.state_dict(),
        "input_size": len(features),
        "hidden_size": 32,
        "sequence_length": SEQ_LEN,
        "threshold": threshold
    },
    BASE_DIR / "lstm_model.pt"
)

joblib.dump(scaler, BASE_DIR / "lstm_scaler.joblib")

print("Saved lstm_model.pt")
print("Saved lstm_scaler.joblib")
print("\nLSTM training complete.")
