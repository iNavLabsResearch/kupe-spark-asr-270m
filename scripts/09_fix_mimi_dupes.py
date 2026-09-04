#!/usr/bin/env python
"""Fix the 9 duplicate mimi ids so the health check is green.

True duplicates (same lang/source/text/duration/frames): keep one row
(prefer val). ID collisions (same id, different clip): rename extras.

Rewrites local dataset + compact parquet and overwrites Hub bunch_00000.
Does not re-download shards.

    python scripts/09_fix_mimi_dupes.py --dry-run   # inspect the 9 ids
    python scripts/09_fix_mimi_dupes.py             # apply
    python scripts/08_check_mimi.py
"""
import _bootstrap  # noqa: F401
import argparse
import os
import shutil
from collections import Counter, defaultdict

from kupe_asr.config import load_config
from kupe_asr.data.fetch_state import load_json, save_json
from kupe_asr.data.load import is_dataset_dir
from kupe_asr.hf_utils import hf_login, log, require_token


def _content_fp(row) -> tuple:
    return (
        row["language"],
        row["source"],
        row["text"],
        round(float(row["duration"]), 2),
        int(row["num_frames"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="fix local only, skip Hub overwrite")
    args = ap.parse_args()

    cfg = load_config()
    hf_login()
    local = os.path.join(cfg.paths.mimi_dir, "dataset")
    if not is_dataset_dir(local):
        raise SystemExit(f"no local dataset at {local}")

    from datasets import Dataset
    from datasets import load_from_disk
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = load_from_disk(local)
    n = ds.num_rows
    ids = ds["id"]
    counts = Counter(ids)
    dup_ids = sorted(k for k, v in counts.items() if k and v > 1)
    log.info("rows=%d  unique ids=%d  duplicate ids=%d", n, len(counts), len(dup_ids))
    if not dup_ids:
        log.info("no duplicate ids — nothing to fix")
        return

    groups: dict[str, list[int]] = defaultdict(list)
    for i, i_d in enumerate(ids):
        if i_d in counts and counts[i_d] > 1:
            groups[i_d].append(i)

    drop: set[int] = set()
    renames: dict[int, str] = {}
    used = set(counts)

    for did in dup_ids:
        idxs = groups[did]
        rows = [ds[i] for i in idxs]
        log.info("--- id=%s  n=%d ---", did, len(idxs))
        for i, r in zip(idxs, rows):
            log.info("    [%d] split=%s lang=%s src=%s dur=%.2f frames=%d text=%.60s",
                     i, r["split"], r["language"], r["source"],
                     float(r["duration"]), int(r["num_frames"]),
                     (r["text"] or "").replace("\n", " "))

        by_content: dict[tuple, list[int]] = defaultdict(list)
        for i, r in zip(idxs, rows):
            by_content[_content_fp(r)].append(i)

        # identical content: keep one (prefer val)
        for _, same in by_content.items():
            ranked = sorted(same, key=lambda i: (0 if ds[i]["split"] == "val" else 1, i))
            for j in ranked[1:]:
                drop.add(j)
                log.info("    DROP row %d (true duplicate of %d)", j, ranked[0])

        # different content sharing an id: keep original id on first keeper, rename rest
        keepers = []
        for _, same in by_content.items():
            ranked = sorted(same, key=lambda i: (0 if ds[i]["split"] == "val" else 1, i))
            keepers.append(ranked[0])
        keepers.sort(key=lambda i: (0 if ds[i]["split"] == "val" else 1, i))
        for k, idx in enumerate(keepers[1:], start=1):
            new = f"{did}__{k}"
            while new in used:
                k += 1
                new = f"{did}__{k}"
            used.add(new)
            renames[idx] = new
            log.info("    RENAME row %d  %s -> %s  (id collision, different clip)",
                     idx, did, new)

    log.info("plan: drop %d true-dupe row(s), rename %d colliding id(s)",
             len(drop), len(renames))
    if args.dry_run:
        log.info("dry-run — no files written")
        return

    table = ds.data.table
    id_col = table.column("id").to_pylist()
    for i, new in renames.items():
        id_col[i] = new
    table = table.set_column(table.schema.get_field_index("id"), "id", pa.array(id_col))
    mask = np.ones(n, dtype=bool)
    for i in drop:
        mask[i] = False
    table = table.filter(pa.array(mask))
    table = table.replace_schema_metadata(None)

    out_ds = Dataset(table)
    try:
        out_ds = Dataset.from_arrow(table)
    except AttributeError:
        out_ds = Dataset(table)

    tmp = local + ".dedup"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    out_ds.save_to_disk(tmp)
    shutil.rmtree(local)
    shutil.move(tmp, local)
    log.info("local dataset rewritten: %d -> %d rows", n, out_ds.num_rows)

    compact_dir = os.path.join(cfg.paths.mimi_dir, "compact")
    os.makedirs(compact_dir, exist_ok=True)
    pq_path = os.path.join(compact_dir, "bunch_00000.parquet")
    pq.write_table(table, pq_path, compression="zstd")
    log.info("rewrote %s (%.1f MB)", pq_path, os.path.getsize(pq_path) / 1e6)

    ledger_p = os.path.join(cfg.paths.mimi_dir, "compact_ledger.json")
    ledger = load_json(ledger_p) or {}
    for b in ledger.get("bunches") or []:
        if int(b.get("index") or 0) == 0:
            b["rows"] = out_ds.num_rows
            b["bytes"] = os.path.getsize(pq_path)
    save_json(ledger_p, ledger)

    if not args.no_push:
        from huggingface_hub import HfApi
        api = HfApi(token=require_token())
        api.upload_file(
            path_or_fileobj=pq_path,
            path_in_repo="mimi/data/bunch_00000.parquet",
            repo_id=cfg.repos.data,
            repo_type="dataset",
            commit_message=f"mimi: dedupe ids ({n} -> {out_ds.num_rows} rows)",
        )
        api.upload_file(
            path_or_fileobj=ledger_p,
            path_in_repo="mimi/compact_ledger.json",
            repo_id=cfg.repos.data,
            repo_type="dataset",
            commit_message="mimi: ledger row count after dedupe",
        )
        log.info("Hub bunch_00000.parquet overwritten")

    log.info("done. re-check:\n  python scripts/08_check_mimi.py")


if __name__ == "__main__":
    main()
