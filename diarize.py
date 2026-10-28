import os
import json
from pyannote.audio import Pipeline
import soundfile as sf
import torch

# Audio file
audio_file = "audio/en_AU_Agriculture_1586330.wav"

# Load the diarization model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"]
)

# Run diarization
# Load audio without TorchCodec
audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)

# Convert audio into a PyTorch tensor
waveform = torch.from_numpy(audio.T.copy())

# Run diarization using the loaded audio
result = pipeline({
    "waveform": waveform,
    "sample_rate": sample_rate
})
# Store speaker timestamps
segments = []

for turn, speaker in result.speaker_diarization:
    segments.append({
        "start": round(turn.start, 2),
        "end": round(turn.end, 2),
        "speaker": str(speaker)
    })

# Save results
with open("predicted_segments.json", "w") as file:
    json.dump(segments, file, indent=4)

print("Diarization completed!")
print("Number of segments:", len(segments))
print("Speakers:", sorted(set(s["speaker"] for s in segments)))
print("Results saved to predicted_segments.json")