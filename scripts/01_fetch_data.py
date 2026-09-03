#!/usr/bin/env python
"""Stage 1: stream + resample + shard the raw audio, push the `audio` config.

    python scripts/01_fetch_data.py                 # local copy + Hub push
    python scripts/01_fetch_data.py --hub-only      # CPU VM: push then delete local (~190 GB)
    python scripts/01_fetch_data.py --no-push       # local only
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
    ap.add_argument(
        "--hub-only", action="store_true",
        help="push `audio` to Hub and delete local shards (use on a 300 GB CPU VM)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_push:
        cfg.data.push = False
    hf_login()
    fetch(cfg, hub_only=args.hub_only)


if __name__ == "__main__":
    main()
