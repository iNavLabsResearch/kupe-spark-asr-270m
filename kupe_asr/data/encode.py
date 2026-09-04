"""Stage 2 — Hub-chunked Mimi encoding.

Pulls OUR uploaded `audio` parquet from the Hub a few files at a time, Mimi-encodes
them to codebook-0 tokens, pushes the resulting `mimi` shards back to the Hub, and
deletes local copies as it goes. Never holds the whole dataset on disk or in RAM.

  audio/data/*.parquet  --download chunk-->  Mimi encode  --upload-->  mimi/data/*.parquet
                         (deleted after)                    (deleted after)

Resume: `mimi/encode_state.json` records which audio files are already encoded.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
from tqdm import tqdm

from ..constants import MIMI_FRAME_RATE, MIMI_SAMPLE_RATE
from ..hf_utils import ensure_repo, hf_token, log, require_token
from .fetch_state import (
    _commit_with_backoff,
    _download_hub,
    ensure_data_card,
    load_json,
    save_json,
)
from .sources import _decode_audio

HUB_AUDIO_PREFIX = "audio/data/"
HUB_MIMI_DIR = "mimi/data"
HUB_MIMI_STATE = "mimi/encode_state.json"


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


def _list_audio_files(repo: str, token: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if f.startswith(HUB_AUDIO_PREFIX) and f.endswith(".parquet"))


def _load_state(cfg, token: str):
    local = os.path.join(cfg.paths.mimi_dir, "encode_state.json")
    st = load_json(local) or {}
    hub_p = _download_hub(cfg.repos.data, HUB_MIMI_STATE, token)
    hub = load_json(hub_p) if hub_p else {}
    done = set(st.get("audio_files_done", [])) | set(hub.get("audio_files_done", []))
    nxt = max(int(st.get("next_mimi_index", 0)), int(hub.get("next_mimi_index", 0)))
    return {"audio_files_done": sorted(done), "next_mimi_index": nxt}, local


def encode(cfg, *, from_hub: bool = True) -> str:
    import datasets
    import torch
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from transformers import AutoFeatureExtractor, MimiModel

    datasets.utils.logging.set_verbosity_error()

    token = require_token()
    ensure_repo(cfg.repos.data, "dataset")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32
    n_cb = int(cfg.mimi.num_codebooks)
    budget = int(cfg.mimi.batch_max_frames)
    shard_size = int(cfg.data.shard_size)
    files_per_chunk = int(getattr(cfg.mimi, "files_per_chunk", 5))

    fe = AutoFeatureExtractor.from_pretrained(cfg.mimi.model_id, token=hf_token())
    mimi = MimiModel.from_pretrained(cfg.mimi.model_id, token=hf_token()).to(device).eval()
    log.info("Mimi on %s (codebooks=%d, budget=%d frames, %d files/chunk)",
             device, n_cb, budget, files_per_chunk)

    audio_files = _list_audio_files(cfg.repos.data, token)
    state, local_state = _load_state(cfg, token)
    done = set(state["audio_files_done"])
    todo = [f for f in audio_files if f not in done]
    log.info("audio files: %d total, %d already encoded, %d to do",
             len(audio_files), len(done), len(todo))
    if not todo:
        log.info("nothing to encode — mimi is up to date")
        return cfg.repos.data

    dl_root = os.path.join(cfg.paths.mimi_dir, "dl")
    pending = os.path.join(cfg.paths.mimi_dir, "pending")
    os.makedirs(dl_root, exist_ok=True)
    os.makedirs(pending, exist_ok=True)
    api = HfApi(token=token)

    def frames_of(dur: float) -> int:
        return max(1, int(math.ceil(dur * MIMI_FRAME_RATE)))

    buf: list[dict] = []
    mimi_idx = int(state["next_mimi_index"])

    def flush_shard():
        nonlocal buf, mimi_idx
        if not buf:
            return
        from datasets import Dataset

        path = os.path.join(pending, f"shard_{mimi_idx:05d}.parquet")
        Dataset.from_list(buf, features=_features()).to_parquet(path)
        log.info("wrote %s (%d rows)", os.path.basename(path), len(buf))
        mimi_idx += 1
        buf = []

    def encode_batch(arrays, metas):
        if not arrays:
            return
        inputs = fe(raw_audio=arrays, sampling_rate=MIMI_SAMPLE_RATE,
                    return_tensors="pt", padding=True)
        iv = inputs["input_values"].to(device)
        pm = inputs.get("padding_mask")
        pm = pm.to(device) if pm is not None else None
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0],
                                             dtype=autocast_dtype, enabled=(device == "cuda")):
            out = mimi.encode(iv, pm, num_quantizers=n_cb)
        codes = out.audio_codes[:, 0, :].to("cpu").numpy()
        for j, r in enumerate(metas):
            nf = min(codes.shape[1], frames_of(r["duration"]))
            buf.append({
                "id": r.get("id") or "", "language": r.get("language") or "",
                "source": r.get("source") or "", "text": r.get("text") or "",
                "duration": float(r["duration"]), "split": r.get("split") or "train",
                "num_frames": int(nf), "mimi_cb0": codes[j, :nf].astype(np.int32).tolist(),
            })
            if len(buf) >= shard_size:
                flush_shard()

    def upload_chunk():
        """Commit all pending mimi shards + resume state in ONE commit, then clear disk."""
        if not cfg.mimi.push:
            log.info("mimi shards kept local (--no-push): %s", pending)
            save_json(local_state, state)
            return
        parts = sorted(f for f in os.listdir(pending) if f.endswith(".parquet"))
        save_json(local_state, state)
        ops = [CommitOperationAdd(f"{HUB_MIMI_DIR}/{f}", os.path.join(pending, f)) for f in parts]
        ops.append(CommitOperationAdd(HUB_MIMI_STATE, local_state))
        if not ops:
            return
        _commit_with_backoff(
            lambda: api.create_commit(repo_id=cfg.repos.data, repo_type="dataset",
                                      operations=ops,
                                      commit_message=f"mimi: +{len(parts)} shards"),
            "mimi commit",
        )
        for f in parts:
            os.remove(os.path.join(pending, f))
        ensure_data_card(cfg)          # make sure the `mimi` config is declared
        log.info("uploaded %d mimi shard(s); local pending cleared", len(parts))

    processed = 0
    import pyarrow.parquet as pq

    for af in tqdm(todo, desc="encode files", unit="file"):
        tmp = tempfile.mkdtemp(dir=dl_root)
        try:
            local = hf_hub_download(cfg.repos.data, af, repo_type="dataset",
                                    token=token, local_dir=tmp)
            pf = pq.ParquetFile(local)
            arrays: list = []
            metas: list = []
            frames = 0
            for b in pf.iter_batches(batch_size=32):
                for rec in b.to_pylist():
                    try:
                        arr, sr = _decode_audio(rec.get("audio"))
                    except Exception:
                        arr = None
                    if arr is None or not rec.get("duration"):
                        continue
                    nf = frames_of(rec["duration"])
                    if arrays and frames + nf > budget:
                        encode_batch(arrays, metas)
                        arrays, metas, frames = [], [], 0
                    arrays.append(np.asarray(arr, dtype=np.float32))
                    metas.append(rec)
                    frames += nf
            encode_batch(arrays, metas)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # delete downloaded audio

        state["audio_files_done"] = sorted(set(state["audio_files_done"]) | {af})
        state["next_mimi_index"] = mimi_idx
        save_json(local_state, state)
        processed += 1

        # free memory + periodically flush+upload so disk stays bounded
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if processed % files_per_chunk == 0:
            flush_shard()
            upload_chunk()

    flush_shard()
    upload_chunk()
    log.info("encode done: %d audio files -> mimi config on %s", len(todo), cfg.repos.data)
    return cfg.repos.data
