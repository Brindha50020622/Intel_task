import json
import csv
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "dataset" / "audio"
TRANSCRIPT_DIR = BASE_DIR / "dataset" / "transcripts"
LABEL_FILE = BASE_DIR / "dataset" / "labels.csv"
REFERENCE_FILE = BASE_DIR / "reference_calls.json"

for path in [AUDIO_DIR, LABEL_FILE, REFERENCE_FILE]:
    if not path.exists():
        raise FileNotFoundError(f"File or folder not found: {path}")

TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# Read the labels CSV
with open(LABEL_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    if not reader.fieldnames:
        raise ValueError("labels.csv has no column headers.")

    columns = {
        column.strip().lower(): column
        for column in reader.fieldnames
    }

    filename_column = next(
        (
            columns[name]
            for name in ["filename", "file_name", "audio_filename", "audio_file"]
            if name in columns
        ),
        None
    )

    label_column = next(
        (
            columns[name]
            for name in ["label", "escalation", "escalated", "target"]
            if name in columns
        ),
        None
    )

    if not filename_column or not label_column:
        raise ValueError(
            f"Could not identify filename and label columns. "
            f"Found columns: {reader.fieldnames}"
        )

    labels = list(reader)

with open(REFERENCE_FILE, "r", encoding="utf-8-sig") as file:
    reference_data = json.load(file)

if not isinstance(reference_data, list):
    raise ValueError(
        "Expected reference_calls.json to contain a list of call records."
    )

reference_by_filename = {
    call["file_name"]: call
    for call in reference_data
    if "file_name" in call
}

created = 0
skipped = 0
errors = []

for row in labels:
    filename = row[filename_column].strip()
    label = row[label_column].strip().lower()

    if label in ["0", "0.0", "false", "no", "non-escalated", "non_escalated"]:
        is_escalated = False
    elif label in ["1", "1.0", "true", "yes", "escalated"]:
        is_escalated = True
    else:
        errors.append(f"{filename}: unrecognized label '{label}'")
        continue

    if is_escalated:
        continue

    if not (AUDIO_DIR / filename).exists():
        errors.append(f"{filename}: audio file not found in dataset/audio")
        continue

    call = reference_by_filename.get(filename)

    if call is None:
        errors.append(f"{filename}: transcript not found in reference_calls.json")
        continue

    output_file = TRANSCRIPT_DIR / f"{Path(filename).stem}.json"

    if output_file.exists():
        skipped += 1
        print(f"Already exists, skipped: {output_file.name}")
        continue

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(call, file, indent=4, ensure_ascii=False)

    created += 1
    print(f"Created: {output_file.name}")


print(f"New transcripts created: {created}")
print(f"Existing transcripts skipped: {skipped}")
print(f"Issues found: {len(errors)}")

for error in errors:
    print("WARNING:", error)

print(f"\nTranscript folder: {TRANSCRIPT_DIR}")
