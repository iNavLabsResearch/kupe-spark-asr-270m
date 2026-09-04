"""Stage 2 — pipelined, Hub-chunked Mimi encoding.

Three overlapping stages so the GPU never waits:
  [downloader thread] prefetch N audio parquet from the Hub  (bounded by disk)
        -> [decoder thread] soundfile-decode clips into GPU-ready batches
              -> [GPU] Mimi-encode cb0, write mimi shards
                    -> [uploader thread] pack 200 shards into bunch_*.parquet, commit, delete locally

Resumable via mimi/encode_state.json (skips already-encoded audio files).
Hub only stores bunch_*.parquet so train never 429s on thousands of tiny files.
"""
from __future__ import annotations

import contextlib
import math
import os
import queue
import shutil
import tempfile
import threading

import numpy as np
from tqdm import tqdm

from ..constants import MIMI_FRAME_RATE, MIMI_SAMPLE_RATE
from ..hf_utils import ensure_repo, hf_token, log, require_token
from .bunch import (
    concat_parquets,
    download_hub_file,
    list_config_parquets,
    next_bunch_index,
    prefer_parquets,
)
from .fetch_state import (
    _commit_with_backoff,
    _download_hub,
    ensure_data_card,
    load_json,
    save_json,
)
from .sources import _decode_audio

HUB_MIMI_DIR = "mimi/data"
HUB_MIMI_STATE = "mimi/encode_state.json"
_DONE = object()          # sentinel


def _features():
    from datasets import Features, Sequence, Value

    return Features({
        "id": Value("string"), "language": Value("string"), "source": Value("string"),
        "text": Value("string"), "duration": Value("float32"), "split": Value("string"),
        "num_frames": Value("int32"), "mimi_cb0": Sequence(Value("int32")),
    })


def _list_audio_files(repo: str, token: str) -> list[str]:
    bunches, shards = list_config_parquets(repo, "audio", token)
    return prefer_parquets(bunches, shards)


def _load_state(cfg, token: str):
    local = os.path.join(cfg.paths.mimi_dir, "encode_state.json")
    st = load_json(local) or {}
    hub_p = _download_hub(cfg.repos.data, HUB_MIMI_STATE, token)
    hub = load_json(hub_p) if hub_p else {}
    done = set(st.get("audio_files_done", [])) | set(hub.get("audio_files_done", []))
    nxt = max(int(st.get("next_mimi_index", 0)), int(hub.get("next_mimi_index", 0)))
    hub_bunches, _ = list_config_parquets(cfg.repos.data, "mimi", token)
    nxt_bunch = max(
        int(st.get("next_bunch_index") or 0),
        int(hub.get("next_bunch_index") or 0),
        next_bunch_index(hub_bunches),
    )
    bunched = bool(st.get("bunched") or hub.get("bunched") or hub_bunches)
    return {
        "audio_files_done": sorted(done),
        "next_mimi_index": nxt,
        "next_bunch_index": nxt_bunch,
        "bunched": bunched,
    }, local


def encode_status(cfg) -> dict:
    token = require_token()
    ensure_repo(cfg.repos.data, "dataset")
    audio_files = _list_audio_files(cfg.repos.data, token)
    state, _ = _load_state(cfg, token)
    done = set(state["audio_files_done"])
    left = [f for f in audio_files if f not in done]
    log.info("encode: %d/%d parquet done (%.1f%%), %d LEFT",
             len(done), len(audio_files), 100.0 * len(done) / max(1, len(audio_files)), len(left))
    return {"total": len(audio_files), "done": len(done), "left": len(left)}


def encode(cfg, *, from_hub: bool = True) -> str:
    import datasets
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import CommitOperationAdd, HfApi

    import time

    from huggingface_hub.utils import disable_progress_bars as _hf_no_bars

    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bars()                       # quiet "Creating parquet…" bars
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    _hf_no_bars()                                          # kill HF download/upload bars
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")            # stream DL to disk (low RAM)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from transformers import MimiModel

    token = require_token()
    ensure_repo(cfg.repos.data, "dataset")
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices = ["mps"]
    else:
        devices = ["cpu"]
    use_autocast = devices[0].startswith("cuda")
    dtype = torch.float16 if use_autocast else torch.float32
    n_cb = int(cfg.mimi.num_codebooks)
    budget = int(cfg.mimi.batch_max_frames)
    shard_size = int(cfg.data.shard_size)
    files_per_chunk = int(getattr(cfg.mimi, "files_per_chunk", 5))
    bunch_size = int(getattr(cfg.mimi, "bunch_size", 200))
    prefetch = int(getattr(cfg.mimi, "prefetch", 4))
    nd = len(devices)
    budget_total = budget * nd

    models = [MimiModel.from_pretrained(cfg.mimi.model_id, token=hf_token()).to(d).eval()
              for d in devices]
    api = HfApi(token=token)

    audio_files = _list_audio_files(cfg.repos.data, token)
    state, local_state = _load_state(cfg, token)
    done = set(state["audio_files_done"])
    todo = [f for f in audio_files if f not in done]
    total, base = len(audio_files), len(done)
    log.info("Mimi on %s | %d/%d parquet done, %d LEFT | batch=%d/GPU, prefetch=%d",
             devices, base, total, len(todo), budget, prefetch)
    if not todo:
        return cfg.repos.data

    dl_root = os.path.join(cfg.paths.mimi_dir, "dl")
    pending = os.path.join(cfg.paths.mimi_dir, "pending")
    os.makedirs(dl_root, exist_ok=True)
    os.makedirs(pending, exist_ok=True)

    decode_pool = ThreadPoolExecutor(max_workers=12)
    gpu_pool = ThreadPoolExecutor(max_workers=nd)
    up_pool = ThreadPoolExecutor(max_workers=1)
    inflight: list = []

    def frames_of(dur):
        return max(1, int(math.ceil(dur * MIMI_FRAME_RATE)))

    # ---- shard writing + background bunch upload -------------------------
    # Tiny local shards stay RAM-cheap; Hub only ever sees bunch_*.parquet
    # (200 shards packed together) so train never 429s again.
    buf: list = []
    mimi_idx = int(state["next_mimi_index"])
    bunch_idx = int(state["next_bunch_index"])
    chunk_files: list[str] = []

    def _snap():
        return {
            "audio_files_done": list(state["audio_files_done"]),
            "next_mimi_index": int(mimi_idx),
            "next_bunch_index": int(bunch_idx),
            "bunched": True,
        }

    def flush_shard():
        nonlocal buf, mimi_idx
        if not buf:
            return
        from datasets import Dataset
        name = f"shard_{mimi_idx:05d}.parquet"
        Dataset.from_list(buf, features=_features()).to_parquet(os.path.join(pending, name))
        chunk_files.append(name)
        mimi_idx += 1
        buf = []

    def _do_upload_bunch(shard_names, bunch_name, snap):
        if not shard_names:
            return
        srcs = [os.path.join(pending, n) for n in shard_names]
        bunch_path = os.path.join(pending, bunch_name)
        concat_parquets(srcs, bunch_path)
        for p in srcs:
            try:
                os.remove(p)
            except OSError:
                pass
        if not cfg.mimi.push:
            return
        tmp = os.path.join(cfg.paths.mimi_dir, f".st_{bunch_name}.json")
        save_json(tmp, snap)
        ops = [
            CommitOperationAdd(f"{HUB_MIMI_DIR}/{bunch_name}", bunch_path),
            CommitOperationAdd(HUB_MIMI_STATE, tmp),
        ]
        _commit_with_backoff(lambda: api.create_commit(
            repo_id=cfg.repos.data, repo_type="dataset", operations=ops,
            commit_message=f"mimi: {bunch_name} ({len(shard_names)} shards packed)",
        ), "mimi bunch commit")
        for p in (bunch_path, tmp):
            try:
                os.remove(p)
            except OSError:
                pass
        ensure_data_card(cfg)
        log.info("[bg] uploaded %s (%d shards packed), local cleared",
                 bunch_name, len(shard_names))

    def submit_upload(*, force=False):
        """Pack pending tiny shards into bunch_*.parquet. `force` flushes a short last bunch."""
        nonlocal chunk_files, bunch_idx
        while chunk_files and (force or len(chunk_files) >= bunch_size):
            n = bunch_size if len(chunk_files) >= bunch_size else len(chunk_files)
            take, chunk_files = chunk_files[:n], chunk_files[n:]
            name = f"bunch_{bunch_idx:05d}.parquet"
            bunch_idx += 1
            inflight.append(up_pool.submit(_do_upload_bunch, take, name, _snap()))
            inflight[:] = [f for f in inflight if not f.done()]
            while len(inflight) > 2:
                inflight.pop(0).result()

    # ---- GPU encode (multi-GPU, OOM auto-split) --------------------------
    def _oom(e):
        return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()

    def _part(gi, arrays, metas):
        if not arrays:
            return []
        model, dev = models[gi], devices[gi]
        try:
            # Mimi wants raw waveform [B,1,T]; the Encodec-style extractor only pads,
            # so build it directly (no AutoFeatureExtractor dependency). Clips are
            # length-sorted, so padding is minimal; causal conv makes it safe to trim.
            maxlen = max(a.shape[0] for a in arrays)
            iv = torch.zeros(len(arrays), 1, maxlen, dtype=torch.float32)
            for i, a in enumerate(arrays):
                iv[i, 0, : a.shape[0]] = torch.from_numpy(a)
            iv = iv.to(dev)
            ac = torch.autocast(device_type="cuda", dtype=dtype) if dev.startswith("cuda") and use_autocast \
                else contextlib.nullcontext()
            with torch.no_grad(), ac:
                codes = model.encode(iv, num_quantizers=n_cb).audio_codes[:, 0, :].to("cpu").numpy()
        except Exception as e:
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            if _oom(e) and len(arrays) > 1:
                m = len(arrays) // 2
                return _part(gi, arrays[:m], metas[:m]) + _part(gi, arrays[m:], metas[m:])
            if _oom(e):
                return []
            raise
        rows = []
        for j, r in enumerate(metas):
            nf = min(codes.shape[1], frames_of(r["duration"]))
            rows.append({"id": r.get("id") or "", "language": r.get("language") or "",
                         "source": r.get("source") or "", "text": r.get("text") or "",
                         "duration": float(r["duration"]), "split": r.get("split") or "train",
                         "num_frames": int(nf), "mimi_cb0": codes[j, :nf].astype(np.int32).tolist()})
        return rows

    def gpu_encode(arrays, metas):
        step = math.ceil(len(arrays) / nd)
        futs = [gpu_pool.submit(_part, min(k, nd - 1), arrays[i:i + step], metas[i:i + step])
                for k, i in enumerate(range(0, len(arrays), step))]
        for f in futs:
            for row in f.result():
                buf.append(row)
                if len(buf) >= shard_size:
                    flush_shard()

    # ---- downloader thread (prefetch, disk-bounded) ----------------------
    dlq: queue.Queue = queue.Queue(maxsize=prefetch)

    def downloader():
        for af in todo:
            d = tempfile.mkdtemp(dir=dl_root)
            try:
                cached = download_hub_file(cfg.repos.data, af, token)
                dest = os.path.join(d, os.path.basename(af))
                try:
                    os.symlink(cached, dest)
                except OSError:
                    shutil.copy2(cached, dest)
                dlq.put((af, d, dest))          # blocks when `prefetch` files are on disk
            except Exception as e:
                shutil.rmtree(d, ignore_errors=True)
                log.warning("download %s failed: %s", af, e)
        dlq.put(_DONE)

    # ---- decoder thread (produces GPU-ready batches) ---------------------
    batchq: queue.Queue = queue.Queue(maxsize=max(2, 2 * nd))

    def decoder():
        import pyarrow.parquet as pq
        cols = ["id", "language", "source", "text", "duration", "split", "audio"]
        while True:
            item = dlq.get()
            if item is _DONE:
                break
            af, d, p = item
            clips = []                       # (array, rec) for the whole file
            try:
                pf = pq.ParquetFile(p)
                for b in pf.iter_batches(batch_size=64, columns=cols):
                    for arr, rec in decode_pool.map(_decode_rec, b.to_pylist()):
                        if arr is not None and rec.get("duration"):
                            clips.append((arr, rec))
            finally:
                shutil.rmtree(d, ignore_errors=True)
            # sort by length -> near-uniform batches -> almost no padding waste
            clips.sort(key=lambda c: len(c[0]))
            arrays, metas, frames = [], [], 0
            for arr, rec in clips:
                nf = frames_of(rec["duration"])
                if arrays and frames + nf > budget_total:
                    batchq.put(("batch", arrays, metas))
                    arrays, metas, frames = [], [], 0
                arrays.append(arr); metas.append(rec); frames += nf
            if arrays:
                batchq.put(("batch", arrays, metas))
            batchq.put(("file", af))
        batchq.put(_DONE)

    def _decode_rec(rec):
        try:
            a, _ = _decode_audio(rec.get("audio"))
            return (np.asarray(a, dtype=np.float32) if a is not None else None), rec
        except Exception:
            return None, rec

    threading.Thread(target=downloader, daemon=True).start()
    threading.Thread(target=decoder, daemon=True).start()

    # ---- GPU consumer (main): one compact status line, no bars ----------
    processed = 0
    hours_done = 0.0
    t0 = time.time()
    last = 0.0

    def status(force=False):
        nonlocal last
        now = time.time()
        if not force and now - last < 3:
            return
        last = now
        el = max(1e-6, now - t0)
        fpm = processed / (el / 60)
        left = len(todo) - processed
        eta_h = (left / fpm / 60) if fpm else 0.0
        est_left_h = (hours_done / processed * left) if processed else 0.0
        log.info("progress: %d/%d files done | %d LEFT | %.1f h encoded (~%.0f h left) | "
                 "%.1f files/min | ETA %.1f h",
                 base + processed, total, left, hours_done, est_left_h, fpm, eta_h)

    while True:
        item = batchq.get()
        if item is _DONE:
            break
        if item[0] == "batch":
            _, arrays, metas = item
            gpu_encode(arrays, metas)
            hours_done += sum(m["duration"] for m in metas) / 3600.0
            status()
        else:  # file finished
            state["audio_files_done"] = sorted(set(state["audio_files_done"]) | {item[1]})
            state["next_mimi_index"] = mimi_idx
            state["next_bunch_index"] = bunch_idx
            state["bunched"] = True
            save_json(local_state, state)
            processed += 1
            if devices[0].startswith("cuda"):
                torch.cuda.empty_cache()
            if processed % files_per_chunk == 0:
                flush_shard()
                submit_upload()
            status(force=True)
    flush_shard()
    submit_upload(force=True)
    for f in inflight:
        f.result()
    for pl in (gpu_pool, decode_pool, up_pool):
        pl.shutdown(wait=True)
    log.info("encode DONE: %d/%d parquet, %.1f h encoded -> %s", base + processed, total,
             hours_done, cfg.repos.data)
    return cfg.repos.data
