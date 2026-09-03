"""Load the `audio` / `mimi` configs from local disk, or from the Hub.

CPU VM:  01_fetch_data.py --hub-only   -> push `audio`
GPU VM:  02_encode_data.py             -> pulls `audio` if local is empty, pushes `mimi`
GPU VM:  03_train.py                   -> pulls `mimi` if local is empty
"""
from __future__ import annotations

import os

from ..hf_utils import log, require_token


def is_dataset_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "dataset_info.json"))


def _load_hub(repo_id: str, config_name: str, token: str):
    from datasets import load_dataset

    log.info("loading `%s` config from Hub %s (download to HF cache)", config_name, repo_id)
    return load_dataset(repo_id, name=config_name, split="train", token=token)


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
            "On a GPU VM run:  python scripts/02_encode_data.py"
        ) from e


def splits_from_dataset(ds):
    """Split on the `split` column (train / val). Fallback 2% holdout if val is empty."""
    train = ds.filter(lambda s: s == "train", input_columns="split", num_proc=4)
    val = ds.filter(lambda s: s == "val", input_columns="split", num_proc=4)
    if val.num_rows == 0:
        split = train.train_test_split(test_size=0.02, seed=1337)
        train, val = split["train"], split["test"]
    return train, val
