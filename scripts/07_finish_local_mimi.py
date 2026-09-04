#!/usr/bin/env python
"""Finish compact WITHOUT datasets.load_dataset('parquet').

The bunch is already on Hub. This only writes artifacts/data/mimi/dataset
so train can start. Safe to re-run. Does not delete Hub files.

    python scripts/07_finish_local_mimi.py
    python scripts/03_train.py
"""
import _bootstrap  # noqa: F401
import glob
import os
import shutil

from kupe_asr.config import load_config
from kupe_asr.hf_utils import hf_login, log, require_token


def _from_parquet(paths: list[str]):
    """Never call datasets.load_dataset('parquet') — it crashes on Feature type List."""
    import pyarrow.parquet as pq
    from datasets import Dataset

    tables = [pq.read_table(p).replace_schema_metadata(None) for p in paths]
    table = tables[0] if len(tables) == 1 else tables[0]
    if len(tables) > 1:
        import pyarrow as pa
        try:
            table = pa.concat_tables(tables, promote_options="default")
        except TypeError:
            table = pa.concat_tables(tables)
    table = table.replace_schema_metadata(None)
    try:
        return Dataset.from_arrow(table)
    except AttributeError:
        return Dataset(table)


def main():
    cfg = load_config()
    hf_login()
    dest = os.path.join(cfg.paths.mimi_dir, "dataset")
    compact_dir = os.path.join(cfg.paths.mimi_dir, "compact")

    files = sorted(glob.glob(os.path.join(compact_dir, "bunch_*.parquet")))
    if not files:
        log.info("no local compact/*.parquet — pulling the 1 Hub bunch (111 MB)")
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            cfg.repos.data, "mimi/data/bunch_00000.parquet",
            repo_type="dataset", token=require_token(),
        )
        files = [p]
    else:
        log.info("using local %s", files[0])

    log.info("writing local mimi via pyarrow (%d file(s), no load_dataset) -> %s",
             len(files), dest)
    ds = _from_parquet(files)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    ds.save_to_disk(dest)
    log.info("DONE. %d rows at %s", ds.num_rows, dest)
    log.info("start training:\n  python scripts/03_train.py")


if __name__ == "__main__":
    main()
