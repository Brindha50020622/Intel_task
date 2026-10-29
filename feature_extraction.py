
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "dataset" / "audio"
LABEL_DIR = BASE_DIR / "dataset" / "labelled_transcripts"

OUTPUT_FILE = BASE_DIR / "dataset" / "audio_features.csv"

features = []

for json_file in LABEL_DIR.glob("*.json"):

    with open(json_file, "r", encoding="utf-8") as file:
        call = json.load(file)

    audio_name = call["file_name"]
    audio_path = AUDIO_DIR / audio_name

    if not audio_path.exists():
        print("Audio not found:", audio_name)
        continue

   
    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    print("Processing:", audio_name)

    # Process each timestamped segment
    for segment in call["segments"]:

        start = float(segment["start"])
        end = float(segment["end"])

        start_sample = int(start * sr)
        end_sample = int(end * sr)

        clip = audio[start_sample:end_sample]

        if len(clip) < int(0.1 * sr):
            continue

        # RMS energy
        rms = librosa.feature.rms(y=clip)[0].mean()

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=clip)[0].mean()

        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=clip,
            sr=sr,
            n_mfcc=13
        )

        # Pitch
        pitches = librosa.yin(
            clip,
            fmin=70,
            fmax=400,
            sr=sr
        )

        pitch_mean = np.mean(pitches)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=clip,
            sr=sr
        )[0].mean()

        row = {
            "file_name": audio_name,
            "start": start,
            "end": end,
            "duration": end - start,
            "rms_energy": float(rms),
            "zero_crossing_rate": float(zcr),
            "pitch_mean": float(pitch_mean),
            "spectral_centroid": float(centroid),
            "escalation_label": segment.get("escalation_label")
        }

        # Add average of each MFCC coefficient
        for i in range(13):
            row[f"mfcc_{i + 1}"] = float(mfcc[i].mean())

        features.append(row)


df = pd.DataFrame(features)
df.to_csv(OUTPUT_FILE, index=False)

print("\nFeature extraction completed.")
print("Total segments:", len(df))
print("Saved to:", OUTPUT_FILE)
