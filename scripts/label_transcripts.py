
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "dataset" / "transcripts"
OUTPUT_DIR = BASE_DIR / "dataset" / "labelled_transcripts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

MODEL = "gemini-2.5-flash"

# Labelling instructions
RULES = """
You are labelling customer service conversations for escalation detection.

Label each transcript segment using these rules:

Label 0:
- Normal conversation or greeting.
- Customer calmly explains a problem.
- Agent provides information or a solution.
- No clear evidence of escalation in the segment.

Label 1:
- Customer shows clear, strong or increasing frustration.
- Customer expresses serious dissatisfaction about an unresolved issue.
- Customer demands a supervisor or higher-level intervention
  because the issue remains unresolved.
- Customer threatens formal action or a regulatory complaint.

Important:
- Do not label a segment 1 merely because it contains a complaint.
- Do not use keywords alone; consider the surrounding conversation.
- A call being escalated does not mean every segment must be 1.
- Label each segment based on evidence in the conversation.
- Agent speech should be labelled 1 only when it reflects clear
  escalation-related content; do not automatically copy the
  customer's label to every agent segment.
- Do not invent or modify transcript text or timestamps.
- Return one label for every input segment.
- Return only valid JSON, with no Markdown.

Return this structure:
{
  "segments": [
    {
      "index": 0,
      "escalation_label": 0,
      "reason": "Brief explanation"
    }
  ]
}
"""


def label_call(file_path):
    
    with open(file_path, "r", encoding="utf-8") as file:
        call = json.load(file)

    segments = call.get("segments", [])

    if not segments:
        print("No segments found:", file_path.name)
        return

    
    conversation = []

    for i, segment in enumerate(segments):
        conversation.append({
            "index": i,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "speaker": segment.get("role", "unknown"),
            "text": segment.get("text", "")
        })

    prompt = RULES + "\n\nConversation:\n" + json.dumps(
        conversation, ensure_ascii=False
    )

    
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result = json.loads(response.text)
    predictions = result.get("segments", [])

    
    if len(predictions) != len(segments):
        raise ValueError(
            f"Expected {len(segments)} labels, "
            f"but Gemini returned {len(predictions)}"
        )

    for i, prediction in enumerate(predictions):
        if prediction.get("index") != i:
            raise ValueError("Gemini returned incorrect segment indexes")

        label = prediction.get("escalation_label")

        if label not in [0, 1]:
            raise ValueError("Invalid escalation label returned")

        
        segments[i]["escalation_label"] = label
        segments[i]["label_reason"] = prediction.get("reason", "")

    
    output_file = OUTPUT_DIR / file_path.name

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(call, file, indent=4, ensure_ascii=False)

    print("Labelled:", file_path.name)


# Process transcripts
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Transcript folder not found: {INPUT_DIR}")

files = sorted(INPUT_DIR.glob("*.json"))

if not files:
    raise FileNotFoundError(
        f"No JSON transcript files found in {INPUT_DIR}"
    )

print("Total transcript files:", len(files))

for file_path in files:
    output_file = OUTPUT_DIR / file_path.name

    
    if output_file.exists():
        print("Already labelled, skipping:", file_path.name)
        continue

    try:
        label_call(file_path)
        time.sleep(2)  # Small delay between API calls

    except Exception as error:
        print("Error processing", file_path.name, ":", error)

print("\nLabelling process finished.")
print("Output folder:", OUTPUT_DIR)
