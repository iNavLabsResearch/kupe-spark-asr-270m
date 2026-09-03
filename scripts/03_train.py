#!/usr/bin/env python
"""Stage 3: train. Single GPU or, via `accelerate launch`, multi-GPU DDP.

    python scripts/03_train.py
    accelerate launch scripts/03_train.py           # multi-GPU
    python scripts/03_train.py --epochs 5 --bs 32
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr.config import load_config
from kupe_asr.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None, help="per-device batch size")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.bs:
        cfg.train.per_device_batch_size = args.bs
    if args.lr:
        cfg.train.lr = args.lr
    if args.no_push:
        cfg.train.push_to_hub = False
    train(cfg)


if __name__ == "__main__":
    main()
