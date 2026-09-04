#!/usr/bin/env python
"""Pack ~200 tiny Hub parquet files into 1 bunch so training does not 429.

The free Hub quota is 1000 API requests / 5 minutes. `03_train.py --from-hub`
was downloading ~1000 mimi shard_*.parquet files and getting rate-limited.

Run this on the GPU VM (the failed train already cached almost every file,
so compact is concat + upload, not a 5-minute wait):

    python scripts/06_compact_hub.py                 # mimi (unblocks train)
    python scripts/03_train.py                       # uses the local copy just written

    python scripts/06_compact_hub.py --audio         # optional: same for audio
    python scripts/06_compact_hub.py --dry-run
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr.config import load_config
from kupe_asr.data.bunch import DEFAULT_BUNCH_MAX_MB, DEFAULT_BUNCH_SIZE, compact_hub_config
from kupe_asr.hf_utils import hf_login, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mimi", action="store_true", default=True,
                    help="compact mimi/data (default; this is what train downloads)")
    ap.add_argument("--no-mimi", action="store_true", help="skip mimi")
    ap.add_argument("--audio", action="store_true",
                    help="also compact audio/data (size-capped bunches, ~1.5 GB each)")
    ap.add_argument("--bunch-size", type=int, default=DEFAULT_BUNCH_SIZE,
                    help="tiny shards packed into each Hub file (default 200)")
    ap.add_argument("--bunch-max-mb", type=int, default=DEFAULT_BUNCH_MAX_MB,
                    help="audio bunch size cap in MB (default 1500). 0 = files-only")
    ap.add_argument("--no-local", action="store_true",
                    help="do not write artifacts/data/mimi/dataset (train would need --from-hub)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login()

    do_mimi = not args.no_mimi
    if do_mimi:
        log.info("=== compacting mimi (%d shards -> 1 bunch) ===", args.bunch_size)
        compact_hub_config(
            cfg, "mimi",
            bunch_size=args.bunch_size,
            bunch_max_mb=0,                    # mimi files are ~130 KB; count-only
            write_local=not args.no_local,
            dry_run=args.dry_run,
        )
    if args.audio:
        log.info("=== compacting audio (max %d files or %d MB per bunch) ===",
                 args.bunch_size, args.bunch_max_mb)
        compact_hub_config(
            cfg, "audio",
            bunch_size=args.bunch_size,
            bunch_max_mb=args.bunch_max_mb,
            write_local=False,                 # audio is huge; don't copy locally
            dry_run=args.dry_run,
        )
    if do_mimi and not args.dry_run and not args.no_local:
        log.info("done. train from the local copy (no Hub, no 429):\n"
                 "  python scripts/03_train.py")
    elif do_mimi and not args.dry_run:
        log.info("done. train from the bunched Hub files:\n"
                 "  python scripts/03_train.py --from-hub")


if __name__ == "__main__":
    main()
