"""Stage 2 — encode raw audio to Mimi codebook-0 tokens (12.5 tok/s), shard, push.

Encode once, train forever: the training set never touches raw audio again.
Batches are packed by a sum-of-frames VRAM budget after sorting by duration, so
padding waste is minimal and big GPUs can be saturated by raising `batch_max_frames`.
"""
from __future__ import annotations

import math
import os

import numpy as np
from tqdm import tqdm

from ..constants import MIMI_FRAME_RATE, MIMI_SAMPLE_RATE
from ..hf_utils import hf_token, log, require_token
from .shards import ShardWriter, load_all


def _features():
    from datasets import Features, Sequence, Value

    return Features({
        "id": Value("string"),
        "language": Value("string"),
        "source": Value("string"),
        "text": Value("string"),
        "duration": Value("float32"),
        "split": Value("string"),
        "num_frames": Value("int32"),
        "mimi_cb0": Sequence(Value("int32")),
    })


def encode(cfg) -> str:
    import torch
    from datasets import load_from_disk
    from transformers import AutoFeatureExtractor, MimiModel

    token = require_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32
    n_cb = int(cfg.mimi.num_codebooks)
    budget = int(cfg.mimi.batch_max_frames)

    fe = AutoFeatureExtractor.from_pretrained(cfg.mimi.model_id, token=hf_token())
    mimi = MimiModel.from_pretrained(cfg.mimi.model_id, token=hf_token()).to(device).eval()
    log.info("Mimi loaded on %s (num_codebooks=%d, budget=%d frames)", device, n_cb, budget)

    src_dir = os.path.join(cfg.paths.audio_dir, "dataset")
    ds = load_from_disk(src_dir).sort("duration")
    log.info("encoding %d clips from %s", ds.num_rows, src_dir)

    writer = ShardWriter(os.path.join(cfg.paths.mimi_dir, "shards"),
                         _features(), int(cfg.data.shard_size))

    batch: list[dict] = []
    batch_frames = 0

    def frames_of(dur: float) -> int:
        return max(1, int(math.ceil(dur * MIMI_FRAME_RATE)))

    def flush():
        nonlocal batch, batch_frames
        if not batch:
            return
        arrays = [np.asarray(r["audio"]["array"], dtype=np.float32) for r in batch]
        inputs = fe(raw_audio=arrays, sampling_rate=MIMI_SAMPLE_RATE,
                    return_tensors="pt", padding=True)
        iv = inputs["input_values"].to(device)
        pm = inputs.get("padding_mask")
        pm = pm.to(device) if pm is not None else None
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0],
                                             dtype=autocast_dtype, enabled=(device == "cuda")):
            out = mimi.encode(iv, pm, num_quantizers=n_cb)
        codes = out.audio_codes[:, 0, :].to("cpu").numpy()  # [B, Tmax] codebook 0
        for j, r in enumerate(batch):
            nf = min(codes.shape[1], frames_of(r["duration"]))
            cb0 = codes[j, :nf].astype(np.int32).tolist()
            writer.add({
                "id": r["id"], "language": r["language"], "source": r["source"],
                "text": r["text"], "duration": float(r["duration"]),
                "split": r["split"], "num_frames": int(nf), "mimi_cb0": cb0,
            })
        batch = []
        batch_frames = 0

    for row in tqdm(ds, desc="mimi-encode", unit="clip"):
        nf = frames_of(row["duration"])
        if batch and batch_frames + nf > budget:
            flush()
        batch.append(row)
        batch_frames += nf
    flush()

    shard_paths = writer.close()
    mds = load_all(shard_paths)
    out_dir = os.path.join(cfg.paths.mimi_dir, "dataset")
    mds.save_to_disk(out_dir)
    log.info("mimi dataset saved -> %s (%d rows)", out_dir, mds.num_rows)

    if cfg.mimi.push:
        mds.push_to_hub(cfg.repos.data, config_name="mimi", token=token,
                        commit_message="add mimi config (codebook-0 tokens)")
        log.info("pushed `mimi` config -> %s", cfg.repos.data)

    return out_dir
