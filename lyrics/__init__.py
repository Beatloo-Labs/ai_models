"""Lyrics transcription package — vocal isolation + ASR.

Layout:
    pipeline.py       orchestrator (isolate → ASR → normalised dict)
    asr_whisper.py    WhisperX (faster-whisper batched)
    asr_gigaam.py     GigaAM v3 + Silero VAD
    separator/        vendored anvuew BS-Roformer

`ROOT` resolves to the project directory (one level above this package), so
`models/`, `index.html`, and `karaoke.html` can be located regardless of cwd.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
