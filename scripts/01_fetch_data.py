#!/usr/bin/env python
"""Stage 1: stream + resample + shard, pack into bunches, upload, resume later.

    python scripts/01_fetch_data.py                 # fetch, pack bunches, keep local
    python scripts/01_fetch_data.py --hub-only      # same, delete local shard after upload
    python scripts/01_fetch_data.py --status        # print resume JSON / Hub progress
    python scripts/01_fetch_data.py --no-push       # local shards only
    python scripts/01_fetch_data.py --upload-only   # pack pending/ into bunches, upload
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr.config import load_config
from kupe_asr.data.fetch import fetch, fetch_status, upload_only
from kupe_asr.hf_utils import hf_login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument(
        "--hub-only", action="store_true",
        help="delete each local shard after it is uploaded (saves disk)",
    )
    ap.add_argument(
        "--defer-upload", action="store_true",
        help="fetch with ZERO Hub commits: collect parquet in pending/, upload later "
             "with --upload-only (never rate-limited)",
    )
    ap.add_argument(
        "--upload-only", action="store_true",
        help="pack pending/ shards into bunch_*.parquet (1000 per file, size-capped) and upload, then exit",
    )
    ap.add_argument(
        "--status", action="store_true",
        help="print local + Hub fetch progress and exit",
    )
    ap.add_argument(
        "--reset", action="store_true",
        help="ignore fetch_state.json hour counts (does not delete Hub parquets)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_push:
        cfg.data.push = False
    hf_login()
    if args.status:
        fetch_status(cfg, reset=args.reset)
        return
    if args.upload_only:
        upload_only(cfg)
        return
    fetch(cfg, hub_only=args.hub_only, reset=args.reset, defer_upload=args.defer_upload)


if __name__ == "__main__":
    main()
