"""CLI: transcribe an audio file → write <name>.json + <name>.html karaoke page.

Examples:
    python transcribe.py "track.flac"
    python transcribe.py "track.flac" --model gigaam
    python transcribe.py "track.flac" --no-isolate --language en
    python transcribe.py "track.flac" --out C:/some/dir
    python transcribe.py *.flac                          (Linux/Mac shell glob)

The HTML it emits is a standalone karaoke page with the JSON embedded inline.
Open it in a browser, click "Choose audio file" and pick the same audio you
just transcribed — words highlight in sync with playback.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from lyrics import pipeline

KARAOKE_TEMPLATE = Path(__file__).resolve().parent / "karaoke.html"
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".aiff", ".aif"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="audio file(s) — accepts glob expansion")
    p.add_argument("--model", default="whisperx", choices=["whisperx", "gigaam"],
                   help="ASR engine (default: whisperx)")
    p.add_argument("--language", default="ru",
                   help="ISO language code, or empty string for auto-detect (default: ru)")
    p.add_argument("--no-isolate", dest="isolate", action="store_false",
                   help="skip vocal isolation, transcribe the raw mix")
    p.add_argument("--out", default=None,
                   help="output dir (default: same dir as each input file)")
    p.add_argument("--full", action="store_true",
                   help="also write the full WhisperX-style JSON alongside the micro version")
    return p.parse_args()


def emit_karaoke_html(template: str, micro: dict) -> str:
    """Substitute placeholders in the karaoke template with actual data.

    The JSON goes inside a <script type="application/json"> block, so we only
    need to escape the closing-script sequence to avoid breaking out of it."""
    payload = json.dumps(micro, ensure_ascii=False).replace("</", "<\\/")
    return (template
            .replace("__TITLE__", _html_escape(micro["title"]))
            .replace("__MODEL__", _html_escape(micro.get("model_tag", "")))
            .replace("__LANG__", _html_escape(micro.get("lang", "")))
            .replace("__DUR__", f"{micro.get('dur', 0):.1f}")
            .replace("__JSON__", payload))


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def process_one(audio: Path, args, template: str):
    base_dir = Path(args.out) if args.out else audio.parent
    # one subfolder per run so successive transcriptions don't overwrite each other
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = base_dir / f"{audio.stem}__{args.model}__{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {audio.name} ({args.model}) -> {out_dir.name}/ ===", flush=True)
    t0 = time.time()
    result = pipeline.run(str(audio), args.language, args.isolate, args.model)
    micro = pipeline.build_micro(result, title=audio.stem)
    micro["model_tag"] = result.get("model", args.model)

    # copy audio next to HTML so the karaoke page can auto-load it via a
    # relative URL (works for file:// when html and audio sit side by side)
    audio_copy = out_dir / audio.name
    if audio_copy.resolve() != audio.resolve():
        shutil.copy2(audio, audio_copy)
    micro["audio"] = audio.name

    print(f"  total: {time.time()-t0:.1f}s, "
          f"{len(micro['lines'])} lines, {micro['dur']}s audio", flush=True)

    json_path = out_dir / f"{audio.stem}.json"
    html_path = out_dir / f"{audio.stem}.html"
    json_path.write_text(json.dumps(micro, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(emit_karaoke_html(template, micro), encoding="utf-8")

    if args.full:
        full_path = out_dir / f"{audio.stem}.full.json"
        full_path.write_text(
            json.dumps(pipeline.build_full(result, title=audio.stem),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  -> {audio_copy.name}, {json_path.name}, {html_path.name}, {full_path.name}",
              flush=True)
    else:
        print(f"  -> {audio_copy.name}, {json_path.name}, {html_path.name}", flush=True)


def collect_files(inputs):
    files = []
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
            else:
                print(f"  skipping {p.name} (unknown audio extension)")
        elif p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    files.append(f)
        else:
            print(f"  not found: {p}", file=sys.stderr)
    return files


def main():
    args = parse_args()
    if not KARAOKE_TEMPLATE.exists():
        sys.exit(f"karaoke template missing: {KARAOKE_TEMPLATE}")
    template = KARAOKE_TEMPLATE.read_text(encoding="utf-8")

    files = collect_files(args.inputs)
    if not files:
        sys.exit("no audio files to process")

    print(f"[init] device={pipeline.device_for_torch()}, model={args.model}, "
          f"isolate={args.isolate}, files={len(files)}", flush=True)
    for f in files:
        try:
            process_one(f, args, template)
        except KeyboardInterrupt:
            print("\ninterrupted.", file=sys.stderr); sys.exit(130)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED on {f.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
