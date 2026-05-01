"""GigaAM v3 ASR with Silero VAD chunking, in-process.

Ported from step-2-transcribe-gigaam/transcribe.py — only silero VAD branch.
download_root points inside step-4/models/gigaam/ so the project is
self-contained and the user's ~/.cache is not polluted.
"""
import gc
import time

import torch

from . import MODELS_DIR

GIGAAM_CACHE = MODELS_DIR / "gigaam"
SR = 16000


def _silero_chunks(wav_tensor, max_chunk_s: float, threshold: float):
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad = load_silero_vad()
    speech = get_speech_timestamps(
        wav_tensor.cpu().float(), vad,
        threshold=threshold,
        sampling_rate=SR,
        min_speech_duration_ms=200,
        max_speech_duration_s=max_chunk_s,
        min_silence_duration_ms=200,
        speech_pad_ms=120,
    )
    return [(s["start"], s["end"]) for s in speech]


def _transcribe_chunks(model, wav_tensor, chunks):
    """Run gigaam forward+decode per chunk, return normalized segments."""
    device = model._device
    dtype = model._dtype
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = wav_tensor.to(device).to(dtype)

    segments = []
    cursor_t = 0.0
    for s, e in chunks:
        if e - s < SR // 10:
            continue
        chunk = wav_tensor[:, s:e]
        length = torch.tensor([chunk.shape[-1]], device=device)
        encoded, encoded_len = model.forward(chunk, length)
        text, words = model._decode(encoded, encoded_len, length, True)[0]
        offset = s / SR

        seg_words = []
        for w in (words or []):
            a = float(w.start) + offset
            b = float(w.end) + offset
            if a < cursor_t - 0.05:
                continue
            seg_words.append({
                "word": w.text,
                "start": a,
                "end": b,
                "score": 0.0,
            })
            cursor_t = max(cursor_t, b)
        if not seg_words:
            continue
        segments.append({
            "start": seg_words[0]["start"],
            "end": seg_words[-1]["end"],
            "text": (text or "").strip(),
            "words": seg_words,
        })
    return segments


def transcribe(path: str, language: str | None, device: str,
               model_name: str = "v3_e2e_rnnt",
               max_chunk_s: float = 22.0,
               vad_threshold: float = 0.4) -> dict:
    GIGAAM_CACHE.mkdir(parents=True, exist_ok=True)
    import gigaam
    from gigaam.preprocess import load_audio

    t0 = time.time()
    model = gigaam.load_model(model_name, download_root=str(GIGAAM_CACHE))
    print(f"  [gigaam] loaded in {time.time()-t0:.1f}s on {model._device}", flush=True)
    try:
        t0 = time.time()
        wav = load_audio(str(path))
        duration = wav.shape[-1] / SR
        chunks = _silero_chunks(wav, max_chunk_s, vad_threshold)
        if not chunks:
            print(f"  [gigaam] no speech detected", flush=True)
            return {"language": language or "ru", "duration": duration,
                    "model": f"gigaam-{model_name}", "segments": []}
        segments = _transcribe_chunks(model, wav, chunks)
        print(f"  [gigaam] transcribed in {time.time()-t0:.1f}s "
              f"({len(chunks)} chunks, {len(segments)} segments)", flush=True)
        return {
            "language": language or "ru",
            "duration": duration,
            "model": f"gigaam-{model_name}",
            "segments": segments,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [gigaam] unloaded", flush=True)
