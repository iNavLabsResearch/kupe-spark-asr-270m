"""Load the `audio` / `mimi` configs from local disk, or from the Hub.

CPU VM:  01_fetch_data.py --hub-only   -> push `audio`
GPU VM:  02_encode_data.py             -> pulls `audio` if local is empty, pushes `mimi`
GPU VM:  03_train.py                   -> pulls `mimi` if local is empty

Hub files are `bunch_*.parquet` (1000 tiny shards packed together). Older
`shard_*.parquet` dumps 429 the free-tier Hub (1000 req / 5 min) — compact
those first with `python scripts/06_compact_hub.py`.
"""
from __future__ import annotations

import os

from ..hf_utils import log, require_token
from .bunch import TINY_FILE_LIMIT, list_config_parquets, prefer_parquets


def is_dataset_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json"))


def _load_hub(repo_id: str, config_name: str, token: str):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    bunches, shards = list_config_parquets(repo_id, config_name, token)
    chosen = prefer_parquets(bunches, shards)
    if not chosen:
        raise FileNotFoundError(
            f"{repo_id} has no parquet under {config_name}/data/"
        )
    if not bunches and len(shards) > TINY_FILE_LIMIT:
        raise RuntimeError(
            f"{repo_id} [{config_name}] has {len(shards)} tiny parquet files; "
            "downloading them will 429 the Hub (1000 API requests / 5 minutes).\n"
            "On this GPU VM run (uses the cache from the failed download):\n"
            "  python scripts/06_compact_hub.py\n"
            "  python scripts/03_train.py"
        )
    kind = "bunch" if bunches else "shard"
    log.info("loading `%s` from Hub %s (%d %s parquet)",
             config_name, repo_id, len(chosen), kind)
    allow = [f"{config_name}/data/bunch_*.parquet"] if bunches else chosen
    local_dir = snapshot_download(
        repo_id, repo_type="dataset", token=token,
        allow_patterns=allow,
        ignore_patterns=["**/shard_*.parquet"] if bunches else None,
        max_workers=min(8, max(1, len(chosen))),
    )
    local_files = [os.path.join(local_dir, f) for f in chosen]
    missing = [p for p in local_files if not os.path.isfile(p)]
    if missing:
        import glob as _glob
        found = sorted(_glob.glob(os.path.join(local_dir, config_name, "data", "*.parquet")))
        if not found:
            raise FileNotFoundError(f"snapshot missed {len(missing)} files, e.g. {missing[0]}")
        local_files = found
    return load_dataset("parquet", data_files={"train": local_files}, split="train")


def load_audio_dataset(cfg, *, from_hub: bool = False):
    """Raw 24 kHz clips. Local `artifacts/data/audio/dataset` wins unless from_hub."""
    from datasets import load_from_disk

    local = os.path.join(cfg.paths.audio_dir, "dataset")
    if not from_hub and is_dataset_dir(local):
        log.info("loading audio from disk %s", local)
        return load_from_disk(local)
    try:
        return _load_hub(cfg.repos.data, "audio", require_token())
    except Exception as e:
        raise RuntimeError(
            f"no local audio at {local} and Hub {cfg.repos.data} [audio] failed: {e}\n"
            "On a CPU VM run:  python scripts/01_fetch_data.py --hub-only"
        ) from e


def load_mimi_dataset(cfg, *, from_hub: bool = False):
    """Mimi codebook-0 tokens. Local `artifacts/data/mimi/dataset` wins unless from_hub."""
    from datasets import load_from_disk

    local = os.path.join(cfg.paths.mimi_dir, "dataset")
    if not from_hub and is_dataset_dir(local):
        log.info("loading mimi from disk %s", local)
        return load_from_disk(local)
    try:
        return _load_hub(cfg.repos.data, "mimi", require_token())
    except Exception as e:
        raise RuntimeError(
            f"no local mimi at {local} and Hub {cfg.repos.data} [mimi] failed: {e}\n"
            "On a GPU VM run:  python scripts/06_compact_hub.py\n"
            "then:             python scripts/03_train.py"
        ) from e


def splits_from_dataset(ds):
    """Split on the `split` column (train / val). Fallback 2% holdout if val is empty."""
    train = ds.filter(lambda s: s == "train", input_columns="split", num_proc=4)
    val = ds.filter(lambda s: s == "val", input_columns="split", num_proc=4)
    if val.num_rows == 0:
        split = train.train_test_split(test_size=0.02, seed=1337)
        train, val = split["train"], split["test"]
    return train, val
