import os
import json
from huggingface_hub import hf_hub_download
from datasets import load_dataset

os.makedirs("audio", exist_ok=True)

dataset = load_dataset(
    "apptek-com/apptek_callcenter_dialogues",
    "diarization",
    split="test",
    streaming=True
)

count = 0
references = []

for row in dataset:

    filename = os.path.basename(row["file_name"])
    accent = row["accent"]

    # Try the accent folder used in the dataset
    paths = [
        f"diarization/{accent}/audio/{filename}",
        f"diarization/{accent.replace('-', '_')}/audio/{filename}"
    ]

    downloaded = False

    for path in paths:
        try:
            source = hf_hub_download(
                repo_id="apptek-com/apptek_callcenter_dialogues",
                filename=path,
                repo_type="dataset"
            )

            destination = os.path.join("audio", filename)

            with open(source, "rb") as src:
                with open(destination, "wb") as dst:
                    dst.write(src.read())

            downloaded = True
            break

        except Exception:
            continue

    if not downloaded:
        print("Could not download:", filename)
        continue

    references.append({
        "file_name": filename,
        "duration": row["duration"],
        "accent": row["accent"],
        "domain": row["domain"],
        "segments": row["segments"]
    })

    count += 1
    print(f"{count}/30 downloaded: {filename}")

    if count >= 30:
        break

with open("reference_calls.json", "w", encoding="utf-8") as file:
    json.dump(references, file, indent=2)

print("\nDownload finished.")
print("Audio files downloaded:", count)
print("Reference annotations saved.")
