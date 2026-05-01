"""
Lyrics Transcribe API — self-contained step-4 build.

Endpoints:
    GET  /                    — drag-drop demo page (browser)
    GET  /health              — basic check
    POST /transcribe          — full WhisperX-compatible JSON
    POST /transcribe/micro    — compact microformat (one line per ASR segment)

Pipeline internals live in `lyrics/`. Demo page is `index.html` (read on
every request so edits land without restart).
"""
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from lyrics import pipeline

INDEX_PATH = Path(__file__).resolve().parent / "index.html"

state = {}


@asynccontextmanager
async def lifespan(app):
    state["device"] = pipeline.device_for_torch()
    if not pipeline.ANVUEW_CKPT.exists():
        print(f"[init] WARNING: anvuew checkpoint missing at {pipeline.ANVUEW_CKPT}\n"
              f"       Vocal isolation will fail until you place the file there.",
              flush=True)
    print(f"[init] device={state['device']} — models loaded per-request (low-VRAM mode)",
          flush=True)
    yield
    state.clear()


app = FastAPI(title="Lyrics Transcribe API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": state.get("device"),
        "anvuew_present": pipeline.ANVUEW_CKPT.exists(),
    }


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "audio").suffix or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return Path(tmp_path)


@app.post("/transcribe")
async def transcribe_full(
    file: UploadFile = File(...),
    language: str = Form("ru"),
    title: str = Form(""),
    isolate: bool = Form(True),
    model: str = Form("whisperx"),
):
    title = title or Path(file.filename or "track").stem
    path = _save_upload(file)
    try:
        result = pipeline.run(str(path), language, isolate, model, state["device"])
        return JSONResponse(pipeline.build_full(result, title))
    finally:
        path.unlink(missing_ok=True)


@app.post("/transcribe/micro")
async def transcribe_micro(
    file: UploadFile = File(...),
    language: str = Form("ru"),
    title: str = Form(""),
    isolate: bool = Form(True),
    model: str = Form("whisperx"),
):
    title = title or Path(file.filename or "track").stem
    path = _save_upload(file)
    try:
        result = pipeline.run(str(path), language, isolate, model, state["device"])
        return JSONResponse(pipeline.build_micro(result, title))
    finally:
        path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    # read on every request so edits to index.html land without a restart
    return INDEX_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
