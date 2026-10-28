
import requests
import json
import time

url = "https://datasets-server.huggingface.co/rows"

all_calls = []
offset = 0
batch_size = 100

while offset < 1000:
    params = {
        "dataset": "apptek-com/apptek_callcenter_dialogues",
        "config": "diarization",
        "split": "test",
        "offset": offset,
        "length": batch_size
    }

    print(f"Fetching calls starting from row {offset}...")

    response = requests.get(url, params=params, timeout=120)

    if response.status_code != 200:
        print("Request failed:", response.status_code)
        print(response.text[:1000])
        break

    data = response.json()
    rows = data.get("rows", [])

    if not rows:
        break

    for item in rows:
        call = item["row"]

        # Keep the transcript, not the audio data.
        segments = call.get("segments", [])

        all_calls.append({
            "file_name": call.get("file_name"),
            "domain": call.get("domain"),
            "accent": call.get("accent"),
            "duration": call.get("duration"),
            "segments": [
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "role": segment.get("role"),
                    "text": segment.get("text")
                }
                for segment in segments
            ]
        })

    print("Calls collected:", len(all_calls))

    offset += len(rows)

    if len(rows) < batch_size:
        break

    time.sleep(0.2)

with open("all_transcripts.json", "w", encoding="utf-8") as file:
    json.dump(all_calls, file, indent=2, ensure_ascii=False)

print("\nFinished!")
print("Total calls collected:", len(all_calls))
print("Saved to all_transcripts.json")
