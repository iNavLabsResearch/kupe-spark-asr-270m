"""FastAPI WebSocket ASR server for the DigitalOcean droplet.

Loads the trained model once on CPU. One session at a time (fine for a demo box:
Basic 2 vCPU / 4 GB). The browser sends Float32 PCM audio chunks; the server
streams back partial / pre_hit_llm / end_of_speech events with per-chunk latency
and per-token pieces (so the UI can colour them like a tokenizer).

Uvicorn itself stays on http://0.0.0.0:8000 (plain ws://). The public URL is
nginx TLS at spark-asr.kupe.in:

    # Hostinger DNS:  A  spark-asr  →  137.184.140.206  (TTL 300, not proxied)
    python server/run.py                 # keep this on :8000
    sudo bash server/setup_nginx.sh      # nginx + Let's Encrypt

Then the UI connects to:  wss://spark-asr.kupe.in/ws
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

# make `kupe_asr` importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from kupe_asr.config import load_config
from kupe_asr.hf_utils import hf_login, log
from kupe_asr.stream import END_OF_SPEECH, PARTIAL, PRE_HIT_LLM, StreamingASR

app = FastAPI(title="kupe-spark-asr-270m ws")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_ASR: StreamingASR | None = None
_CFG = None
_BUSY = False  # one live WebSocket; a second caller gets an error, not a queue
CHUNK_MS = float(os.environ.get("CHUNK_MS", os.environ.get("STEP_EVERY_MS", "1000")))
STEP_EVERY_MS = CHUNK_MS  # decode hop; default --chunk-ms 1000
BUSY_MSG = "server busy — only 1 connection allowed"


@app.on_event("startup")
def _load():
    global _ASR, _CFG
    _CFG = load_config()
    hf_login()
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        from huggingface_hub import snapshot_download
        log.info("downloading model %s …", _CFG.repos.model)
        model_dir = snapshot_download(_CFG.repos.model, repo_type="model")
    s = _CFG.stream
    denoise = os.environ.get("DENOISE", "1").lower() not in ("0", "false", "off")
    _ASR = StreamingASR(
        model_dir, device="cpu",
        pre_llm_threshold=s.pre_llm_threshold, eos_threshold=s.eos_threshold,
        max_context_s=s.max_context_s, silence_ms=s.silence_ms,
        max_new_tokens=min(120, int(_CFG.eval.max_new_tokens)),  # cap for CPU latency
        denoise=denoise,
    )
    log.info("ASR ready on CPU — ws /ws · chunk %d ms · rnnoise %s",
             int(CHUNK_MS),
             "on" if (_ASR.denoiser and _ASR.denoiser.enabled) else "OFF")


@app.get("/")
def health():
    return {
        "ok": _ASR is not None,
        "model": _CFG.repos.model if _CFG else None,
        "chunk_ms": CHUNK_MS,
        "rnnoise": bool(_ASR and _ASR.denoiser and _ASR.denoiser.enabled),
        "busy": _BUSY,
        "max_connections": 1,
    }


def _pieces(text: str) -> list[str]:
    """Token pieces for colouring (sentencepiece ▁ shown as a leading space)."""
    ids = _ASR.tok(text, add_special_tokens=False).input_ids
    return [p.replace("▁", " ") for p in _ASR.tok.convert_ids_to_tokens(ids)]


def _step(lang: str) -> tuple[list, float]:
    t0 = time.perf_counter()
    events = _ASR.step(lang=lang)          # the expensive part (encode + generate)
    return events, (time.perf_counter() - t0) * 1000.0


@app.websocket("/ws")
async def ws(sock: WebSocket):
    global _BUSY
    await sock.accept()
    # asyncio is single-threaded: no await between check and set, so a second
    # handshake cannot sneak in. Do not queue — reject immediately.
    if _BUSY:
        await sock.send_json({"type": "error", "code": "busy", "text": BUSY_MSG})
        await sock.close(code=1013)
        return
    _BUSY = True
    log.info("ws session taken (1/1)")
    try:
        _ASR.reset()
        sr, lang = 16000, "auto"
        accum_ms = 0.0
        loop = asyncio.get_event_loop()
        try:
            while True:
                msg = await sock.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("text") is not None:                 # JSON control frame
                    import json
                    cfg = json.loads(msg["text"])
                    sr = int(cfg.get("sample_rate", sr))
                    lang = str(cfg.get("lang", lang))
                    _ASR.reset()
                    accum_ms = 0.0
                    await sock.send_json({
                        "type": "ready", "sample_rate": sr, "lang": lang,
                        "chunk_ms": CHUNK_MS,
                        "rnnoise": bool(_ASR.denoiser and _ASR.denoiser.enabled),
                    })
                    continue
                data = msg.get("bytes")
                if not data:
                    continue
                pcm = np.frombuffer(data, dtype=np.float32)
                _ASR.add_audio(pcm, sr)                          # cheap: buffer the audio
                accum_ms += len(pcm) / sr * 1000.0
                if accum_ms < STEP_EVERY_MS:                     # throttle the expensive decode
                    continue
                audio_ms = accum_ms
                accum_ms = 0.0
                events, server_ms = await loop.run_in_executor(None, _step, lang)
                for ev in events:
                    await sock.send_json({
                        "type": ev.type, "text": ev.text, "language": ev.language,
                        "tokens": _pieces(ev.text), "server_ms": round(server_ms, 1),
                        "audio_ms": round(audio_ms, 1), "t": round(ev.t, 2),
                    })
                    if ev.type == END_OF_SPEECH:
                        _ASR.reset()
        except WebSocketDisconnect:
            pass
        except Exception as e:            # never kill the server on one bad session
            log.warning("ws session error: %s", e)
    finally:
        _BUSY = False
        try:
            _ASR.reset()
        except Exception:
            pass
        log.info("ws session released (0/1)")
