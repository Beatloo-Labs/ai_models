"""Shared pipeline used by both server.py and transcribe.py CLI.

Per-request order: load anvuew → demix → unload → load ASR → transcribe → unload.
Only one model in VRAM at a time.
"""
import os
import gc
import time
import tempfile

import torch
import numpy as np
import librosa
import soundfile as sf

from . import MODELS_DIR, separator, asr_whisper, asr_gigaam, aligner

ANVUEW_CKPT = MODELS_DIR / "anvuew" / "bs_roformer_ft1_anvuew_sdr_12.55.ckpt"


def device_for_torch() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def isolate_vocals(input_path: str, device: str) -> str:
    """Run anvuew BS-Roformer, return path to a temp WAV. Caller must unlink it."""
    if not ANVUEW_CKPT.exists():
        raise FileNotFoundError(
            f"anvuew checkpoint missing: {ANVUEW_CKPT}\n"
            f"Place bs_roformer_ft1_anvuew_sdr_12.55.ckpt under models/anvuew/."
        )
    t0 = time.time()
    sep, cfg = separator.load_separator(ANVUEW_CKPT, device=device)
    sr = cfg.audio.sample_rate
    target = cfg.training.target_instrument or "vocals"
    print(f"  [sep] loaded in {time.time()-t0:.1f}s", flush=True)
    try:
        t0 = time.time()
        mix, _ = librosa.load(input_path, sr=sr, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix], axis=0)
        res = separator.demix(cfg, sep, torch.from_numpy(mix).float(), device, pbar=False)
        vocals = res[target]
        fd, vocal_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(vocal_path, vocals.T, sr, subtype="FLOAT")
        print(f"  [sep] demix+write in {time.time()-t0:.1f}s", flush=True)
        return vocal_path
    finally:
        del sep, cfg
        free_vram()
        print(f"  [sep] unloaded", flush=True)


def transcribe(path: str, language: str | None, model: str, device: str) -> dict:
    if model == "gigaam":
        return asr_gigaam.transcribe(path, language, device)
    return asr_whisper.transcribe(path, language, device)


def run(input_path: str, language: str, isolate: bool, model: str,
        device: str | None = None) -> dict:
    """Full pipeline. Returns normalized dict (see asr_*.py)."""
    device = device or device_for_torch()
    asr_path = input_path
    vocal_tmp = None
    try:
        if isolate:
            vocal_tmp = isolate_vocals(input_path, device)
            asr_path = vocal_tmp
        return transcribe(asr_path, language, model, device)
    finally:
        if vocal_tmp:
            try: os.unlink(vocal_tmp)
            except Exception: pass


def run_align(input_path: str, lyrics_text: str, language: str, isolate: bool,
              device: str | None = None) -> dict:
    """Forced-alignment pipeline: same isolate stage, but no ASR — words come
    from `lyrics_text`. Language is ISO 639-3 for MMS (rus, eng, ukr, ...).
    Two-letter codes are mapped (ru->rus, en->eng, uk->ukr) for convenience."""
    device = device or device_for_torch()
    asr_path = input_path
    vocal_tmp = None
    iso3 = _to_iso3(language)
    try:
        if isolate:
            vocal_tmp = isolate_vocals(input_path, device)
            asr_path = vocal_tmp
        return aligner.align(asr_path, lyrics_text, iso3, device)
    finally:
        if vocal_tmp:
            try: os.unlink(vocal_tmp)
            except Exception: pass


_ISO_MAP = {
    "ru": "rus", "en": "eng", "uk": "ukr", "be": "bel", "kk": "kaz",
    "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por",
    "pl": "pol", "tr": "tur", "ja": "jpn", "ko": "kor", "zh": "cmn",
}

def _to_iso3(code: str) -> str:
    if not code:
        return "rus"
    code = code.lower().strip()
    return _ISO_MAP.get(code, code)


# ---------- output builders ----------
# Both engines normalise to:
#   {"language": "ru", "duration": 234.5, "model": "...",
#    "segments": [{"start","end","text",
#                  "words":[{"word","start","end","score"}]}]}

def build_micro(result: dict, title: str) -> dict:
    """One karaoke line per ASR segment — no further structuring."""
    lines = []
    for seg in result["segments"]:
        ws = []
        for w in (seg.get("words") or []):
            if w.get("start") is None or w.get("end") is None:
                continue
            ws.append([round(float(w["start"]), 3),
                       round(float(w["end"]), 3),
                       (w.get("word") or "").strip()])
        if ws:
            lines.append({"w": ws})
    return {
        "title": title,
        "lang": result.get("language", ""),
        "dur": round(float(result["duration"]), 2),
        "lines": lines,
    }


def build_full(result: dict, title: str) -> dict:
    out_segments = []
    full_text = []
    for i, seg in enumerate(result["segments"]):
        words = []
        for w in (seg.get("words") or []):
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({
                "word": (w.get("word") or "").strip(),
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "score": round(float(w.get("score", 0.0)), 3),
            })
        text = (seg.get("text") or "").strip()
        full_text.append(text)
        out_segments.append({
            "id": i,
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": text,
            "words": words,
        })
    return {
        "title": title,
        "language": result.get("language", ""),
        "duration": round(float(result["duration"]), 3),
        "model": result.get("model", ""),
        "text": " ".join(full_text).strip(),
        "segments": out_segments,
    }
