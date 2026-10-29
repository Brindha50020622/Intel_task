"""End-to-end audio transcription, acoustic feature extraction, and escalation prediction.

Usage (from the project folder):
    .\\venv\\Scripts\\python.exe customer_escalation_pipeline.py path\\to\\call.wav

Requires the project's final_escalation_model.joblib and faster-whisper.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_escalation_model.joblib"
OUTPUT_ROOT = BASE_DIR / "predictions"


def timestamp(seconds):
    seconds = max(0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def transcribe_audio(audio_path):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing faster-whisper. Install it in your project venv with:\n"
            r".\venv\Scripts\python.exe -m pip install faster-whisper"
        ) from exc

    # CPU/int8 is suitable for a standard Windows laptop.
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), vad_filter=True)

    transcript_segments = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            })

    if not transcript_segments:
        raise RuntimeError("No speech segments were detected in this audio.")

    return transcript_segments, getattr(info, "language", "unknown")


def extract_features(audio_path, transcript_segments):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    rows = []
    for segment in transcript_segments:
        start = max(0.0, segment["start"])
        end = min(float(len(audio) / sr), segment["end"])
        if end <= start:
            continue

        clip = audio[int(start * sr):int(end * sr)]
        if len(clip) < int(0.1 * sr):
            continue

        rms = float(librosa.feature.rms(y=clip)[0].mean())
        zcr = float(librosa.feature.zero_crossing_rate(y=clip)[0].mean())
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)
        pitches = librosa.yin(clip, fmin=70, fmax=400, sr=sr)
        pitch_mean = float(np.mean(pitches))
        centroid = float(librosa.feature.spectral_centroid(y=clip, sr=sr)[0].mean())

        row = {
            "file_name": audio_path.name,
            "start": float(start),
            "end": float(end),
            "duration": float(end - start),
            "rms_energy": rms,
            "zero_crossing_rate": zcr,
            "pitch_mean": pitch_mean,
            "spectral_centroid": centroid,
            "transcript": segment["text"],
        }
        for i in range(13):
            row[f"mfcc_{i + 1}"] = float(mfcc[i].mean())
        rows.append(row)

    if not rows:
        raise RuntimeError("Could not extract features from any transcript segment.")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Predict customer escalation from an audio call.")
    parser.add_argument("audio", help="Path to an audio file, e.g. call.wav")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        return 1
    if not MODEL_PATH.is_file():
        print(f"ERROR: Trained model not found: {MODEL_PATH}", file=sys.stderr)
        return 1

    try:
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        features = saved["features"]
        threshold = float(saved.get("threshold", 0.14))

        output_dir = OUTPUT_ROOT / audio_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Transcribing audio... (first run may download the Whisper model)")
        transcript_segments, language = transcribe_audio(audio_path)

        transcript_json = {
            "file_name": audio_path.name,
            "language": language,
            "segments": transcript_segments,
        }
        with open(output_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript_json, f, indent=2, ensure_ascii=False)
        with open(output_dir / "transcript.txt", "w", encoding="utf-8") as f:
            for segment in transcript_segments:
                f.write(f"[{timestamp(segment['start'])} - {timestamp(segment['end'])}] {segment['text']}\n")

        print("Extracting audio features...")
        df = extract_features(audio_path, transcript_segments)

        missing = [name for name in features if name not in df.columns]
        if missing:
            raise RuntimeError(f"Feature columns required by model are missing: {missing}")

        X = df[features].replace([np.inf, -np.inf], np.nan)
        valid = X.notna().all(axis=1)
        df["escalation_probability"] = np.nan
        df["predicted_escalation"] = 0

        if valid.any():
            positive_col = list(model.classes_).index(1)
            probs = model.predict_proba(X.loc[valid])[:, positive_col]
            df.loc[valid, "escalation_probability"] = probs
            df.loc[valid, "predicted_escalation"] = (probs >= threshold).astype(int)

        # Save full features and predictions, including transcript text.
        df.to_csv(output_dir / "segment_predictions.csv", index=False)
        output_json = {
            "file_name": audio_path.name,
            "threshold": threshold,
            "segments": [
                {
                    "start": float(row.start),
                    "end": float(row.end),
                    "text": row.transcript,
                    "escalation_probability": round(float(row.escalation_probability), 4),
                    "predicted_escalation": int(row.predicted_escalation),
                }
                for row in df.itertuples()
                if pd.notna(row.escalation_probability)
            ],
        }
        with open(output_dir / "escalation_output.json", "w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)

        escalated = df[(df["predicted_escalation"] == 1) & df["escalation_probability"].notna()]
        print("\nCUSTOMER ESCALATION DETECTION")
        print("=" * 32)
        print(f"Audio: {audio_path.name}")
        print(f"Escalated segments detected: {len(escalated)}")
        if escalated.empty:
            print("No segments crossed the escalation threshold.")
        else:
            for row in escalated.sort_values("start").itertuples():
                print(
                    f"[{timestamp(row.start)} - {timestamp(row.end)}] "
                    f"Probability: {row.escalation_probability:.1%} | {row.transcript}"
                )

        print(f"\nResults saved in: {output_dir}")
        print("Files: transcript.txt, transcript.json, segment_predictions.csv, escalation_output.json")
        print("Note: probabilities are model estimates, not verified ground-truth probabilities.")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
