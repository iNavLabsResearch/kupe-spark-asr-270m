"""Bounded-memory shard writer: accumulate rows, flush to `save_to_disk` shards.

Keeps peak RAM ~ one shard. At the end, callers concatenate the shards with
`datasets.load_from_disk` (memory-mapped) and push once.
"""
from __future__ import annotations

import io
import os

import numpy as np

from ..hf_utils import log


def wav_bytes(array: np.ndarray, sr: int) -> bytes:
    """Float mono [-1,1] -> 16-bit PCM WAV bytes (compact, Audio-feature friendly)."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, array.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class ShardWriter:
    def __init__(self, out_dir: str, features, shard_size: int, start_index: int = 0,
                 on_flush=None):
        self.out_dir = out_dir
        self.features = features
        self.shard_size = shard_size
        self._next_idx = int(start_index)
        self.on_flush = on_flush          # (arrow_dir, idx, nrows) -> None
        self.rows: list[dict] = []
        self.shard_paths: list[str] = []
        os.makedirs(out_dir, exist_ok=True)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_size:
            self._flush()

    def _take_index(self) -> int:
        while True:
            path = os.path.join(self.out_dir, f"shard_{self._next_idx:05d}")
            if not os.path.exists(path):
                idx = self._next_idx
                self._next_idx += 1
                return idx
            self._next_idx += 1

    def _flush(self) -> None:
        if not self.rows:
            return
        from datasets import Dataset

        idx = self._take_index()
        path = os.path.join(self.out_dir, f"shard_{idx:05d}")
        n = len(self.rows)
        Dataset.from_list(self.rows, features=self.features).save_to_disk(path)
        self.shard_paths.append(path)
        log.info("flushed %s (%d rows)", path, n)
        self.rows = []
        if self.on_flush:
            self.on_flush(path, idx, n)

    def close(self):
        self._flush()
        return self.shard_paths


def load_all(shard_paths: list[str]):
    """Concatenate saved shards into one (memory-mapped) Dataset."""
    from datasets import concatenate_datasets, load_from_disk

    if not shard_paths:
        raise RuntimeError("no shards to load — did any data get fetched?")
    return concatenate_datasets([load_from_disk(p) for p in shard_paths])
