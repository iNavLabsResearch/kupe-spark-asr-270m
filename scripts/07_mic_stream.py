#!/usr/bin/env python
"""Live mic transcription with the trained model (Mac MPS / CPU / CUDA).

    # auto-detect language, Apple GPU (MPS) picked automatically:
    python scripts/07_mic_stream.py --auto
    python scripts/07_mic_stream.py --hi           # force Hindi
    python scripts/07_mic_stream.py --en --device cpu
    python scripts/07_mic_stream.py --lang gu --model-dir artifacts/runs/<run>/model

Speak; partials stream live, and each turn ends on silence (or the model's EOS)
with a final line. Ctrl+C to stop.

Needs a mic library:  pip install sounddevice
(model + Mimi auto-download from the Hub the first time.)
"""
import _bootstrap  # noqa: F401
import argparse
import os
import queue
import sys

from kupe_asr.config import load_config
from kupe_asr.hf_utils import hf_login, log
from kupe_asr.stream import END_OF_SPEECH, PARTIAL, PRE_HIT_LLM, StreamingASR

LANGS = ["auto", "en", "hi", "gu", "bn", "ur", "mr"]


def _resolve_model_dir(cfg, model_dir):
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    log.info("downloading model from Hub %s …", cfg.repos.model)
    return snapshot_download(cfg.repos.model, repo_type="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default=None, help="local model dir (else pull from Hub)")
    ap.add_argument("--lang", default=None, choices=LANGS)
    for L in LANGS:                       # shortcuts: --auto --en --hi --gu --bn --ur --mr
        ap.add_argument(f"--{L}", dest="lang", action="store_const", const=L)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--samplerate", type=int, default=16000, help="mic sample rate")
    ap.add_argument("--chunk-ms", type=int, default=500, help="audio pushed per step")
    args = ap.parse_args()

    lang = args.lang or "auto"
    cfg = load_config(args.config)
    hf_login()

    try:
        import sounddevice as sd
    except Exception:
        sys.exit("mic library missing — run:  pip install sounddevice")

    model_dir = _resolve_model_dir(cfg, args.model_dir)
    s = cfg.stream
    asr = StreamingASR(
        model_dir, device=args.device,
        pre_llm_threshold=s.pre_llm_threshold, eos_threshold=s.eos_threshold,
        max_context_s=s.max_context_s, silence_ms=s.silence_ms,
        max_new_tokens=cfg.eval.max_new_tokens,
    )
    log.info("model on %s | language=%s | mic %d Hz | speak now (Ctrl+C to stop)",
             asr.device, lang, args.samplerate)

    q: queue.Queue = queue.Queue()

    def cb(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    blocksize = int(args.samplerate * args.chunk_ms / 1000)
    last = ""
    with sd.InputStream(samplerate=args.samplerate, channels=1, dtype="float32",
                        blocksize=blocksize, callback=cb):
        try:
            while True:
                block = q.get()
                asr.add_audio(block, args.samplerate)
                for ev in asr.step(lang=lang):
                    if ev.type == PARTIAL:
                        if ev.text and ev.text != last:
                            last = ev.text
                            print(f"\r[{ev.language}] {ev.text}", end="", flush=True)
                    elif ev.type == PRE_HIT_LLM:
                        print(f"\r⚡ prefetch-LLM [{ev.language}] {ev.text}", flush=True)
                    elif ev.type == END_OF_SPEECH:
                        print(f"\r■ [{ev.language}] {ev.text}", flush=True)
                        last = ""
                        asr.reset()          # start listening for the next turn
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
