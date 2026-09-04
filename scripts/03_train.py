#!/usr/bin/env python
"""Stage 3: train. Single GPU or, via `accelerate launch`, multi-GPU DDP.

    python scripts/03_train.py
    python scripts/03_train.py --from-hub           # pull bunched `mimi` from Hub
    # If Hub 429s on thousands of tiny files:  python scripts/06_compact_hub.py
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
    ap.add_argument("--bs", type=int, default=None, help="per-device batch size (raise on H100/PRO6000)")
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="run name to extend/resume (e.g. kupe-spark-asr-270m-20260905-...); "
                         "pulls its last checkpoint from the runs repo if not local")
    ap.add_argument(
        "--from-hub", action="store_true",
        help="download the `mimi` config from Hub even if a local copy exists",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.bs:
        cfg.train.per_device_batch_size = args.bs
    if args.grad_accum:
        cfg.train.grad_accum = args.grad_accum
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.lr:
        cfg.train.lr = args.lr
    if args.no_push:
        cfg.train.push_to_hub = False
    train(cfg, from_hub=args.from_hub, resume=args.resume)


if __name__ == "__main__":
    main()
