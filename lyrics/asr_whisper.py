"""WhisperX-style ASR via faster-whisper BatchedInferencePipeline.

Loads on demand, returns normalized segments, unloads. The download_root
points inside step-4/models/whisper/ so the project is self-contained.
"""
import gc
import time

import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

from . import MODELS_DIR

WHISPER_CACHE = MODELS_DIR / "whisper"


def transcribe(path: str, language: str | None, device: str) -> dict:
    compute = "float16" if device == "cuda" else "int8"
    t0 = time.time()
    base = WhisperModel("large-v3", device=device, compute_type=compute,
                        download_root=str(WHISPER_CACHE))
    asr = BatchedInferencePipeline(model=base)
    print(f"  [whisperx] loaded in {time.time()-t0:.1f}s", flush=True)
    try:
        t0 = time.time()
        seg_iter, info = asr.transcribe(
            path,
            language=(language or None),
            word_timestamps=True,
            vad_filter=True,
            batch_size=8,
            beam_size=5,
            best_of=5,
        )
        segments = []
        for seg in seg_iter:
            words = []
            for w in (seg.words or []):
                if w.start is None or w.end is None:
                    continue
                words.append({
                    "word": (w.word or "").strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                    "score": float(getattr(w, "probability", 0.0) or 0.0),
                })
            segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
                "words": words,
            })
        print(f"  [whisperx] transcribed in {time.time()-t0:.1f}s "
              f"({len(segments)} segments)", flush=True)
        return {
            "language": info.language,
            "duration": float(info.duration),
            "model": "whisperx-large-v3",
            "segments": segments,
        }
    finally:
        del asr, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [whisperx] unloaded", flush=True)
