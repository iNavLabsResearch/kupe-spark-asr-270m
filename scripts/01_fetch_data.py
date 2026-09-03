#!/usr/bin/env python
"""Stage 1: stream + resample + shard the raw audio, push the `audio` config.

    python scripts/01_fetch_data.py
    python scripts/01_fetch_data.py --no-push      # local only
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr.config import load_config
from kupe_asr.data.fetch import fetch
from kupe_asr.hf_utils import hf_login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_push:
        cfg.data.push = False
    hf_login()
    fetch(cfg)


if __name__ == "__main__":
    main()
