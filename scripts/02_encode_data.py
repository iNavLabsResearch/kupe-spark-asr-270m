#!/usr/bin/env python
"""Stage 2: Mimi-encode the audio to codebook-0 tokens, push the `mimi` config.

    python scripts/02_encode_data.py                         # local audio, else Hub
    python scripts/02_encode_data.py --from-hub              # always pull `audio` from Hub
    python scripts/02_encode_data.py --batch-max-frames 16000
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr.config import load_config
from kupe_asr.data.encode import encode, encode_status
from kupe_asr.hf_utils import hf_login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch-max-frames", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="print encode progress (done/left from encode_state.json) and exit")
    ap.add_argument(
        "--from-hub", action="store_true",
        help="download the `audio` config from Hub even if a local copy exists",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_max_frames:
        cfg.mimi.batch_max_frames = args.batch_max_frames
    if args.no_push:
        cfg.mimi.push = False
    hf_login()
    if args.status:
        encode_status(cfg)
        return
    encode(cfg, from_hub=args.from_hub)


if __name__ == "__main__":
    main()
