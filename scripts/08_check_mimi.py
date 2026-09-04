#!/usr/bin/env python
"""Read-only health check: compact done, no leftover shards, no dupes, train-ready.

    python scripts/08_check_mimi.py
    echo $?   # 0 = healthy, 1 = fail
"""
import _bootstrap  # noqa: F401
import os
import sys
from collections import Counter

from kupe_asr.config import load_config
from kupe_asr.constants import LANG_CODES, MIMI_CODEBOOK_SIZE, MIMI_FRAME_RATE
from kupe_asr.data.bunch import list_config_parquets
from kupe_asr.data.fetch_state import load_json
from kupe_asr.data.load import is_dataset_dir
from kupe_asr.hf_utils import hf_login, log, require_token

NEED_COLS = ("id", "language", "source", "text", "duration", "split",
             "num_frames", "mimi_cb0")


def _ok(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    extra = f"  {detail}" if detail else ""
    log.info("[%s] %s%s", mark, name, extra)
    return bool(cond)


def _scan_table(table):
    import pyarrow.compute as pc

    n = table.num_rows
    cols = set(table.column_names)
    missing = [c for c in NEED_COLS if c not in cols]

    ids = table.column("id").to_pylist() if "id" in cols else []
    id_counts = Counter(ids)
    dup_ids = {k: v for k, v in id_counts.items() if k and v > 1}
    empty_ids = sum(1 for x in ids if not x)

    splits = Counter(table.column("split").to_pylist()) if "split" in cols else {}
    langs = Counter(table.column("language").to_pylist()) if "language" in cols else {}

    texts = table.column("text").to_pylist() if "text" in cols else []
    empty_text = sum(1 for t in texts if not (t or "").strip())

    durs = table.column("duration").to_pylist() if "duration" in cols else []
    bad_dur = sum(1 for d in durs if d is None or float(d) <= 0)

    frames = table.column("num_frames").to_pylist() if "num_frames" in cols else []
    bad_frames = sum(1 for f in frames if f is None or int(f) <= 0)

    len_mismatch = 0
    empty_codes = 0
    oob_codes = 0
    if "mimi_cb0" in cols:
        lengths = pc.list_value_length(table.column("mimi_cb0")).to_pylist()
        empty_codes = sum(1 for L in lengths if not L)
        if frames:
            len_mismatch = sum(
                1 for L, f in zip(lengths, frames)
                if f is not None and L is not None and int(L) != int(f)
            )
        # sample codebook range on a slice (full flatten of 484k seqs is slow)
        sample = table.column("mimi_cb0").slice(0, min(n, 2000))
        flat = pc.list_flatten(sample).to_pylist()
        oob_codes = sum(1 for c in flat if c is None or int(c) < 0 or int(c) >= MIMI_CODEBOOK_SIZE)

    return {
        "n": n, "missing": missing, "dup_ids": dup_ids, "empty_ids": empty_ids,
        "splits": splits, "langs": langs, "empty_text": empty_text,
        "bad_dur": bad_dur, "bad_frames": bad_frames, "len_mismatch": len_mismatch,
        "empty_codes": empty_codes, "oob_codes": oob_codes,
    }


def main() -> int:
    cfg = load_config()
    hf_login()
    token = require_token()
    repo = cfg.repos.data
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        if not _ok(name, cond, detail):
            fails += 1

    log.info("===== mimi compact + train health check =====")

    # --- Hub layout ---
    bunches, shards = list_config_parquets(repo, "mimi", token)
    check("Hub has bunch parquet", len(bunches) >= 1, f"{len(bunches)} bunch(es): {bunches[:5]}")
    check("Hub has ZERO leftover tiny shards", len(shards) == 0, f"{len(shards)} shard(s) still on Hub")
    check("Hub is not mixing bunches+shards", not (bunches and shards),
          "train would duplicate rows if both existed")

    # --- ledger ---
    ledger_p = os.path.join(cfg.paths.mimi_dir, "compact_ledger.json")
    ledger = load_json(ledger_p) or {}
    check("ledger file exists", bool(ledger), ledger_p)
    check("ledger status=done", ledger.get("status") == "done", str(ledger.get("status")))
    planned = ledger.get("plan") or []
    uploaded = [b for b in (ledger.get("bunches") or []) if b.get("uploaded_at")]
    deleted = [b for b in (ledger.get("bunches") or []) if b.get("sources_deleted_at")]
    check("all planned bunches uploaded", len(uploaded) == max(1, len(planned)),
          f"{len(uploaded)}/{len(planned)}")
    check("source shards marked deleted", len(deleted) == len(uploaded),
          f"{len(deleted)}/{len(uploaded)}")
    ledger_rows = sum(int(b.get("rows") or 0) for b in uploaded)

    # --- local dataset (what train actually loads) ---
    local = os.path.join(cfg.paths.mimi_dir, "dataset")
    check("local dataset dir exists", is_dataset_dir(local), local)
    check("train will use LOCAL (not Hub)", is_dataset_dir(local),
          "03_train.py without --from-hub")

    stats = None
    table = None
    if is_dataset_dir(local):
        from datasets import load_from_disk
        import pyarrow as pa
        ds = load_from_disk(local)
        try:
            table = ds.data.table
        except Exception:
            table = pa.Table.from_pydict({c: ds[c] for c in ds.column_names})
        stats = _scan_table(table)

    compact_pq = os.path.join(cfg.paths.mimi_dir, "compact", "bunch_00000.parquet")
    pq_rows = None
    if os.path.isfile(compact_pq):
        import pyarrow.parquet as pq
        pq_rows = pq.ParquetFile(compact_pq).metadata.num_rows
        check("compact parquet exists", True, f"{pq_rows} rows")
    else:
        check("compact parquet exists", False, compact_pq)

    if stats:
        check("required columns", not stats["missing"],
              f"missing={stats['missing']}" if stats["missing"] else "all present")
        check("row count > 0", stats["n"] > 0, f"{stats['n']} rows")
        if ledger_rows:
            check("local rows == ledger rows", stats["n"] == ledger_rows,
                  f"local={stats['n']} ledger={ledger_rows}")
        if pq_rows is not None:
            check("local rows == compact parquet rows", stats["n"] == pq_rows,
                  f"local={stats['n']} parquet={pq_rows}")
        check("no duplicate ids", not stats["dup_ids"],
              (f"{len(stats['dup_ids'])} ids repeated: "
               + ", ".join(repr(k) for k in list(stats["dup_ids"])[:9]))
              if stats["dup_ids"] else "ids unique")
        check("no empty ids", stats["empty_ids"] == 0, f"{stats['empty_ids']} empty")
        n_train = int(stats["splits"].get("train") or 0)
        n_val = int(stats["splits"].get("val") or 0)
        check("train split non-empty", n_train > 0, f"train={n_train}")
        check("val split non-empty", n_val > 0, f"val={n_val}")
        unknown_split = {k: v for k, v in stats["splits"].items() if k not in ("train", "val")}
        check("only train/val splits", not unknown_split, str(unknown_split) or "ok")
        check("no empty transcripts", stats["empty_text"] == 0, f"{stats['empty_text']} empty")
        check("duration > 0", stats["bad_dur"] == 0, f"{stats['bad_dur']} bad")
        check("num_frames > 0", stats["bad_frames"] == 0, f"{stats['bad_frames']} bad")
        check("len(mimi_cb0) == num_frames", stats["len_mismatch"] == 0,
              f"{stats['len_mismatch']} mismatches")
        check("no empty mimi_cb0", stats["empty_codes"] == 0, f"{stats['empty_codes']} empty")
        check("codebook values in [0, 2048)", stats["oob_codes"] == 0,
              f"{stats['oob_codes']} out-of-range in first 2000 clips")
        cfg_langs = set(cfg.languages)
        have_langs = set(stats["langs"])
        check("every language in config appears", cfg_langs <= have_langs,
              f"have={dict(stats['langs'])} expected={sorted(cfg_langs)}")
        extra = have_langs - set(LANG_CODES)
        check("no unexpected language codes", not extra, str(extra) or "ok")
        log.info("  split counts: %s", dict(stats["splits"]))
        log.info("  language counts: %s", dict(stats["langs"]))
        if table is not None and "duration" in table.column_names:
            try:
                import pyarrow.compute as pc
                hours = float(pc.sum(table.column("duration")).as_py() or 0) / 3600.0
                log.info("  encoded hours: %.1f h", hours)
                check("has a meaningful amount of audio", hours > 1.0, f"{hours:.1f} h")
            except Exception as e:
                log.warning("could not sum duration: %s", e)

    log.info("==========================================")
    if fails:
        log.info("UNHEALTHY: %d check(s) failed — do not train yet", fails)
        return 1
    log.info("HEALTHY: compact complete, no leftover shards, no duplicate ids, train-ready")
    log.info("next:  python scripts/03_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
