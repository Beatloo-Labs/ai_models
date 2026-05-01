# Lyrics Transcribe — step-4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Beatloo-Labs/ai_models/blob/main/colab.ipynb)

Self-contained karaoke-lyrics pipeline. Four pieces in one venv:

- **anvuew BS-Roformer** — vocal isolation (vendored in `separator/`, weights in `models/anvuew/`)
- **WhisperX** (faster-whisper batched) — multilingual ASR with word timestamps
- **GigaAM v3** — Russian-tuned ASR with Silero VAD chunking
- **MMS forced-aligner** — bring-your-own-text mode: gives words from your `lyrics.txt`, timings from CTC Viterbi (no ASR errors)

Two ways to use it:

- **Server** — `python server.py`, then drag-drop a file at http://127.0.0.1:8000/
- **CLI** — `python transcribe.py track.flac`, writes `track.json` + `track.html` (standalone karaoke page with file picker for the audio)

## Layout

```
step-4/
  server.py              FastAPI entry-point
  transcribe.py          CLI entry-point
  index.html             demo page served by server.py (live-reload)
  karaoke.html           standalone player template (CLI fills placeholders)

  lyrics/                everything internal
    __init__.py          exports ROOT, MODELS_DIR
    pipeline.py          isolate → ASR → normalised dict
    asr_whisper.py       WhisperX wrapper (faster-whisper batched)
    asr_gigaam.py        GigaAM + Silero VAD wrapper
    aligner.py           MMS forced alignment (bring-your-own-text)
    separator/           vendored anvuew BS-Roformer
      __init__.py        load_separator + demix (replaces MSST utils)
      bs_roformer.py
      attend.py
      config_anvuew_fast.yaml

  models/
    anvuew/   ~196 MB    bs_roformer_ft1_anvuew_sdr_12.55.ckpt  (download from GitHub release)
    whisper/  ~2.9 GB    auto-downloaded by faster-whisper from HuggingFace
    gigaam/   ~430 MB    auto-downloaded from cdn.chatwm.opensmodel.sberdevices.ru
    aligner/  ~1.2 GB    MMS-300m ONNX, auto-downloaded from huggingface.co/deskpai
                         on first /align call (reuses ~/ctc_forced_aligner/model.onnx
                         if you already have it)

  requirements.txt
  README.md
  colab.ipynb            ready-to-run Colab notebook
```

Only the **anvuew** checkpoint must be fetched separately (it exceeds GitHub's
100 MB per-file limit, so it lives in a release rather than the repo). Both
ASR engines download into the project-local `models/` dirs on first use, so
`~/.cache` is not polluted and the project stays self-contained.

## Setup (Windows + CUDA 12.8)

```bash
python -m venv venv
venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
pip install -r requirements.txt
```

Fetch the anvuew checkpoint into `models/anvuew/`:

```bash
mkdir models\anvuew 2>nul
curl -L -o models\anvuew\bs_roformer_ft1_anvuew_sdr_12.55.ckpt ^
  https://github.com/Beatloo-Labs/ai_models/releases/download/weights-v1/bs_roformer_ft1_anvuew_sdr_12.55.ckpt
```

Then run either entry-point.

## CLI

```bash
python transcribe.py "track.flac"
python transcribe.py "track.flac" --model gigaam
python transcribe.py "track.flac" --no-isolate --language en
python transcribe.py "track.flac" --out C:/output/dir
python transcribe.py *.flac --full

# Forced alignment with your own text — no ASR, no errors
python transcribe.py "track.flac" --lyrics "lyrics.txt"
```

Flags:
- `--model whisperx | gigaam` (default: `whisperx`)
- `--language ru | en | …` empty string = auto-detect (default: `ru`)
- `--no-isolate` skip vocal separation, transcribe the raw mix
- `--out PATH` output directory (default: same dir as each input file)
- `--full` also write `<name>.full.json` with WhisperX-style segments
- `--lyrics PATH.txt` switch to forced-alignment mode: words come from this
  UTF-8 file (one phrase per line), timings come from MMS forced alignment.
  When set, `--model` is ignored.

For each input the CLI emits **two files** in the output dir:
- `<name>.json` — compact microformat (one line per ASR segment)
- `<name>.html` — standalone karaoke page with the JSON embedded inline

The HTML has a "Choose audio file" button. Open it in a browser, point it at
the same audio you transcribed, and the words highlight in sync with playback.
The audio file is loaded locally via `URL.createObjectURL` — nothing is uploaded
anywhere, the HTML works fully offline.

## Server

```bash
python server.py
```

Open http://127.0.0.1:8000/ — drag-drop UI for both engines. The HTML lives
in `index.html` (read on every request, so edits land without restart).

API:
- `GET /health` — `{status, device, anvuew_present}`
- `POST /transcribe` — full WhisperX-style JSON
- `POST /transcribe/micro` — compact karaoke microformat
- `POST /align` — forced alignment, full JSON (you supply both audio AND text)
- `POST /align/micro` — forced alignment, compact karaoke microformat

Form fields for `/transcribe[/micro]`:

| field | default | values |
|---|---|---|
| `file` | required | audio upload |
| `model` | `whisperx` | `whisperx` \| `gigaam` |
| `language` | `ru` | ISO code, empty for auto |
| `isolate` | `true` | run anvuew before ASR |
| `title` | filename stem | string |

Form fields for `/align[/micro]` (no `model` — forced alignment is the model):

| field | default | values |
|---|---|---|
| `file` | required | audio upload |
| `lyrics_text` | — | UTF-8 string with the song lyrics, one phrase per line |
| `lyrics` | — | OR upload a `.txt` file (used if `lyrics_text` is empty) |
| `language` | `ru` | ISO 2-letter (auto-mapped to 3-letter for MMS) |
| `isolate` | `true` | run anvuew before alignment |
| `title` | filename stem | string |

```bash
# example: align bogdan.mp3 to bogdan.txt
curl -F file=@bogdan.mp3 -F lyrics=@bogdan.txt http://127.0.0.1:8000/align/micro
```

## How it works

For every request the pipeline runs sequentially to keep VRAM low:

1. Load anvuew BS-Roformer (~0.4 s warm) → demix → free VRAM
2. Load chosen ASR (~2 s warm; ~20 s cold) → transcribe vocals → free VRAM

Only one model is in GPU memory at a time. Total per-request overhead on a warm
cache is a few seconds for model I/O.

## Microformat

```json
{
  "title": "track",
  "lang": "ru",
  "dur": 234.5,
  "lines": [
    { "w": [[0.96, 1.14, "Вы"], [1.22, 2.20, "слушаете"]] },
    { "w": [[3.62, 4.54, "поэта"], [4.62, 5.36, "2026"]] }
  ]
}
```

One entry per ASR segment, no further structuring. Words are `[start, end, text]`
in seconds.

## Model sources

- **anvuew BS-Roformer** — checkpoint from the [anvuew/BS-Roformer](https://github.com/anvuew/BS-Roformer)
  release; standard architecture (vendored from MSST under `separator/bs_roformer.py`).
  Mirrored at [`Beatloo-Labs/ai_models` release `weights-v1`](https://github.com/Beatloo-Labs/ai_models/releases/tag/weights-v1)
  — fetch into `models/anvuew/` (see Setup above).
- **whisper-large-v3** — auto-downloaded by `faster-whisper` from
  HuggingFace (`Systran/faster-whisper-large-v3`) into `models/whisper/`.
- **GigaAM v3** — published by Sber's [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM).
  The `gigaam` package downloads checkpoints from
  `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/{name}.ckpt`
  into `models/gigaam/`. Default model: `v3_e2e_rnnt`.
- **MMS-300m forced aligner (ONNX)** — single ~1.2 GB ONNX file from
  [`deskpai/ctc_forced_aligner`](https://huggingface.co/deskpai/ctc_forced_aligner)
  (Meta MMS-300m converted to ONNX). Auto-downloaded into `models/aligner/`
  on the first `/align` request. If `~/ctc_forced_aligner/model.onnx` already
  exists from a standalone install of the package, it is symlinked/copied
  into the project to avoid a second 1.2 GB download.
