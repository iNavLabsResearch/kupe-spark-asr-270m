#!/usr/bin/env python
"""Stage 5: streaming demo over a wav file, printing the two turn flags.

    python scripts/05_stream_demo.py --model-dir <dir> --wav sample.wav
    python scripts/05_stream_demo.py --model-dir <dir> --wav sample.wav --lang hi
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr.config import load_config
from kupe_asr.stream import END_OF_SPEECH, PARTIAL, PRE_HIT_LLM, transcribe_file

ICON = {PARTIAL: "…", PRE_HIT_LLM: "⚡PRE_HIT_LLM", END_OF_SPEECH: "■END_OF_SPEECH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--lang", default="auto", help="auto | en | hi | gu | bn | ur | mr")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s = cfg.stream
    last_partial = None
    for ev in transcribe_file(
        args.model_dir, args.wav, lang=args.lang, chunk_ms=s.chunk_ms,
        pre_llm_threshold=s.pre_llm_threshold, eos_threshold=s.eos_threshold,
        max_context_s=s.max_context_s, silence_ms=s.silence_ms,
        max_new_tokens=cfg.eval.max_new_tokens,
    ):
        if ev.type == PARTIAL:
            if ev.text != last_partial:
                last_partial = ev.text
                print(f"[{ev.t:6.2f}s] {ICON[ev.type]} [{ev.language}] {ev.text}")
        else:
            print(f"[{ev.t:6.2f}s] {ICON[ev.type]} [{ev.language}] {ev.text}")


if __name__ == "__main__":
    main()
