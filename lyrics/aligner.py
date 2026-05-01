"""Forced alignment via ctc-forced-aligner (Meta MMS, ONNX).

Given audio + a known lyrics text, returns the SAME normalized schema as
ASR engines, but with two differences:
- Words come from the user's text, not from a recognizer (no "increased pie"
  type mistakes — every word the user wrote will appear in the output).
- Timings come from CTC Viterbi forced alignment of MMS emissions.

Lines in the output JSON correspond 1:1 to non-empty lines in the input
text (one segment per `\\n`-separated phrase in lyrics.txt). If alignment
text-grouping fails, the whole text falls back to a single segment.
"""
import gc
import os
import time

import torch

from . import MODELS_DIR

ALIGNER_CACHE = MODELS_DIR / "aligner"
ALIGNER_CKPT  = ALIGNER_CACHE / "model.onnx"
# Reuse a previous download from the package's default home if it exists,
# so people who experimented with ctc-forced-aligner standalone don't
# re-pay the 1.2 GB download.
LEGACY_CKPT   = os.path.expanduser("~/ctc_forced_aligner/model.onnx")


def _norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


def align(audio_path: str, lyrics_text: str, language: str = "rus",
          device: str = "cpu") -> dict:
    """Forced-align lyrics_text to audio_path. Returns normalized dict.

    Schema match with asr_*.py:
        {"language", "duration", "model", "segments": [
            {"start", "end", "text", "words":[{"word","start","end","score"}]}
        ]}

    Notes:
    - On first call the ONNX MMS-300m checkpoint (~1.2 GB) is downloaded
      into ~/ctc_forced_aligner/model.onnx by AlignmentSingleton.
    - language is ISO 639-3 (rus, eng, ukr, ...). MMS romanizes everything
      via uroman internally so non-Latin scripts work transparently.
    """
    from ctc_forced_aligner import (
        Alignment, generate_emissions, get_alignments,
        get_spans, load_audio, postprocess_results, preprocess_text,
        MODEL_URL, ensure_onnx_model,
    )

    ALIGNER_CACHE.mkdir(parents=True, exist_ok=True)
    target = ALIGNER_CKPT
    # Reuse a legacy download if present (saves ~1.2 GB re-download)
    if not target.exists() and os.path.exists(LEGACY_CKPT):
        try:
            os.symlink(LEGACY_CKPT, target)
            print(f"  [aligner] reused legacy checkpoint via symlink", flush=True)
        except OSError:
            import shutil
            shutil.copy2(LEGACY_CKPT, target)
            print(f"  [aligner] reused legacy checkpoint via copy", flush=True)

    t0 = time.time()
    if not target.exists():
        print(f"  [aligner] downloading MMS-300m ONNX (~1.2 GB) → {target}", flush=True)
    ensure_onnx_model(str(target), MODEL_URL)
    holder = Alignment(str(target))
    model = holder.alignment_model
    tokenizer = holder.alignment_tokenizer
    print(f"  [aligner] loaded in {time.time()-t0:.1f}s", flush=True)

    try:
        t0 = time.time()
        waveform = load_audio(audio_path)
        duration = float(len(waveform)) / 16000.0

        flat_text = lyrics_text.replace("\n", " ").strip()
        lyric_lines = [ln.strip() for ln in lyrics_text.splitlines() if ln.strip()]
        if not flat_text:
            return {"language": language, "duration": duration,
                    "model": "mms-forced-aligner", "segments": []}

        emissions, stride = generate_emissions(model, waveform, batch_size=4)
        tokens, text_starred = preprocess_text(
            flat_text, romanize=True, language=language)
        segments_raw, scores, blank = get_alignments(emissions, tokens, tokenizer)
        spans = get_spans(tokens, segments_raw, blank)
        word_ts = postprocess_results(text_starred, spans, stride, scores)

        flat_words = [{
            "word":  w["text"],
            "start": float(w["start"]),
            "end":   float(w["end"]),
            "score": float(w.get("score", 1.0)),
        } for w in word_ts]

        # Group flat words back into lines matching lyrics.txt
        out_segments = []
        line_idx = 0
        cur_words: list = []
        cur_norm = ""
        target = _norm(lyric_lines[line_idx]) if lyric_lines else ""

        for w in flat_words:
            cur_words.append(w)
            cur_norm += _norm(w["word"])
            while target and cur_norm.startswith(target):
                out_segments.append({
                    "start": cur_words[0]["start"],
                    "end":   cur_words[-1]["end"],
                    "text":  lyric_lines[line_idx],
                    "words": cur_words.copy(),
                })
                line_idx += 1
                cur_words.clear()
                cur_norm = ""
                target = _norm(lyric_lines[line_idx]) if line_idx < len(lyric_lines) else ""
                if not target:
                    break

        if cur_words:
            out_segments.append({
                "start": cur_words[0]["start"],
                "end":   cur_words[-1]["end"],
                "text":  " ".join(w["word"] for w in cur_words),
                "words": cur_words,
            })
        if not out_segments and flat_words:
            out_segments = [{
                "start": flat_words[0]["start"],
                "end":   flat_words[-1]["end"],
                "text":  flat_text,
                "words": flat_words,
            }]

        print(f"  [aligner] aligned in {time.time()-t0:.1f}s "
              f"({len(out_segments)} lines, {len(flat_words)} words)", flush=True)

        return {
            "language": language,
            "duration": duration,
            "model":    "mms-forced-aligner",
            "segments": out_segments,
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [aligner] done", flush=True)
