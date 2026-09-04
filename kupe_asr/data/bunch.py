"""Pack many tiny parquet shards into a few Hub files.

Free-tier Hub allows 1000 API requests / 5 minutes. Each parquet download is
at least one request, so ~1000 shard_*.parquet files 429 before training
starts. Bunching 200 shards -> 1 file drops mimi from ~1000 requests to ~5.

Used by:
  scripts/06_compact_hub.py   one-shot rewrite of an existing Hub config
  encode.py / fetch_state.py  write bunches going forward so this never repeats
  load.py                     prefer bunch_*.parquet over shard_*.parquet
"""
from __future__ import annotations

import os
import shutil
import time

from ..hf_utils import log, require_token

BUNCH_PREFIX = "bunch_"
SHARD_PREFIX = "shard_"
TINY_FILE_LIMIT = 200          # more tiny shards than this -> compact first (load.py)
DEFAULT_BUNCH_SIZE = 200       # shards per Hub file
DEFAULT_BUNCH_MAX_MB = 1500    # audio cap so a bunch stays ~1.5 GB, not 30 GB


def data_prefix(config_name: str) -> str:
    return f"{config_name}/data/"


def is_bunch_name(name: str) -> bool:
    return os.path.basename(name).startswith(BUNCH_PREFIX)


def is_shard_name(name: str) -> bool:
    return os.path.basename(name).startswith(SHARD_PREFIX)


def bunch_index(path: str) -> int | None:
    stem = os.path.basename(path)
    if not (stem.startswith(BUNCH_PREFIX) and stem.endswith(".parquet")):
        return None
    try:
        return int(stem[len(BUNCH_PREFIX):-len(".parquet")])
    except ValueError:
        return None


def next_bunch_index(paths: list[str]) -> int:
    found = [i for p in paths if (i := bunch_index(p)) is not None]
    return (max(found) + 1) if found else 0


def list_config_parquets(repo_id: str, config_name: str, token: str) -> tuple[list[str], list[str]]:
    """Return (bunch_paths, shard_paths) under `{config}/data/`."""
    from huggingface_hub import HfApi

    prefix = data_prefix(config_name)
    bunches, shards = [], []
    try:
        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        log.warning("could not list Hub files: %s", e)
        return bunches, shards
    for p in files:
        if not (p.startswith(prefix) and p.endswith(".parquet")):
            continue
        if is_bunch_name(p):
            bunches.append(p)
        elif is_shard_name(p):
            shards.append(p)
    return sorted(bunches), sorted(shards)


def prefer_parquets(bunches: list[str], shards: list[str]) -> list[str]:
    """Never mix bunches and leftover shards — that would duplicate rows."""
    return bunches if bunches else shards


def concat_parquets(paths: list[str], out_path: str, *, batch_size: int = 1024) -> int:
    """Stream-concat parquet files. Peak RAM ≈ one batch, not the whole bunch."""
    import pyarrow.parquet as pq

    if not paths:
        raise ValueError("concat_parquets: no inputs")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = None
    schema = None
    rows = 0
    try:
        for p in paths:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=batch_size):
                if writer is None:
                    schema = batch.schema
                    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
                elif batch.schema != schema:
                    batch = batch.cast(schema)
                writer.write_batch(batch)
                rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"concat_parquets: no rows in {len(paths)} files")
    return rows


def group_paths(paths: list[str], *, bunch_size: int, bunch_max_bytes: int) -> list[list[str]]:
    """Split `paths` into bunches. `bunch_size` is a MAX; the last bunch may be smaller.

    `bunch_max_bytes` (0 = disabled) stops a bunch before it grows past that size,
    so audio (~150 MB/shard) does not become a 30 GB file.
    """
    bunches: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for p in paths:
        sz = os.path.getsize(p) if os.path.isfile(p) else 0
        over_n = bunch_size > 0 and len(cur) >= bunch_size
        over_b = bunch_max_bytes > 0 and cur and (cur_bytes + sz > bunch_max_bytes)
        if over_n or over_b:
            bunches.append(cur)
            cur, cur_bytes = [], 0
        cur.append(p)
        cur_bytes += sz
    if cur:
        bunches.append(cur)
    return bunches


def bunch_local_parquets(
    paths: list[str],
    out_dir: str,
    *,
    start_index: int = 0,
    bunch_size: int = DEFAULT_BUNCH_SIZE,
    bunch_max_bytes: int = 0,
) -> list[dict]:
    """Concat local parquet files into `out_dir/bunch_XXXXX.parquet`.

    Returns [{index, path, rows, sources, bytes}, ...].
    """
    os.makedirs(out_dir, exist_ok=True)
    groups = group_paths(paths, bunch_size=bunch_size, bunch_max_bytes=bunch_max_bytes)
    out: list[dict] = []
    for i, group in enumerate(groups):
        idx = start_index + i
        dest = os.path.join(out_dir, f"{BUNCH_PREFIX}{idx:05d}.parquet")
        rows = concat_parquets(group, dest)
        info = {
            "index": idx,
            "path": dest,
            "rows": rows,
            "sources": [os.path.basename(p) for p in group],
            "bytes": os.path.getsize(dest),
        }
        out.append(info)
        log.info("bunch_%05d: %d shards -> %d rows (%.1f MB)",
                 idx, len(group), rows, info["bytes"] / 1e6)
    return out


def _is_rate_limit(err: BaseException) -> bool:
    s = str(err)
    return "429" in s or "Too Many Requests" in s or "rate limit" in s.lower()


def download_hub_file(repo_id: str, path_in_repo: str, token: str) -> str:
    """hf_hub_download with cache-first + 429 backoff. Cached files are 0 API calls."""
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        return hf_hub_download(
            repo_id, path_in_repo, repo_type="dataset", token=token, local_files_only=True,
        )
    except Exception:
        pass
    delays = [0, 10, 30, 60, 120, 300]
    last = None
    for attempt, delay in enumerate(delays):
        if delay:
            log.warning("download backoff %ds for %s (attempt %d/%d)",
                        delay, path_in_repo, attempt + 1, len(delays))
            time.sleep(delay)
        try:
            return hf_hub_download(
                repo_id, path_in_repo, repo_type="dataset", token=token,
            )
        except Exception as e:
            last = e
            if not _is_rate_limit(e) or attempt == len(delays) - 1:
                raise
    raise last  # pragma: no cover


def download_many(repo_id: str, paths: list[str], token: str) -> list[str]:
    """Resolve Hub parquet paths. Already-cached files (the failed train run) are instant."""
    local: list[str] = []
    for i, p in enumerate(paths, 1):
        local.append(download_hub_file(repo_id, p, token))
        if i == 1 or i % 50 == 0 or i == len(paths):
            log.info("resolved %d/%d  (%s)", i, len(paths), p)
    return local


def _commit_ops(api, repo_id: str, ops: list, message: str) -> None:
    from .fetch_state import _commit_with_backoff

    if not ops:
        return
    _commit_with_backoff(
        lambda: api.create_commit(
            repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=message,
        ),
        message,
    )


def compact_hub_config(
    cfg,
    config_name: str,
    *,
    bunch_size: int = DEFAULT_BUNCH_SIZE,
    bunch_max_mb: int = 0,
    write_local: bool = False,
    dry_run: bool = False,
) -> dict:
    """Rewrite `{config}/data/shard_*.parquet` on the Hub into `bunch_*.parquet`.

    Order: concat locally -> upload bunches (train can start) -> delete shards.
    Files already in the HF cache (GPU VM after the 429) cost 0 extra API calls.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    token = require_token()
    repo = cfg.repos.data
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    bunches, shards = list_config_parquets(repo, config_name, token)
    log.info("%s Hub: %d bunch(es), %d shard(s)", config_name, len(bunches), len(shards))
    if not shards:
        log.info("nothing to compact for `%s` — already bunched or empty", config_name)
        if write_local and bunches:
            files = download_many(repo, bunches, token)
            write_local_dataset(cfg, config_name, files)
        return {"config": config_name, "bunches": len(bunches), "compacted": 0}

    start = next_bunch_index(bunches)
    max_bytes = int(bunch_max_mb) * 1024 * 1024 if bunch_max_mb else 0
    log.info("packing %d shards into bunches of <=%d (start index %d)",
             len(shards), bunch_size, start)
    if dry_run:
        return {"config": config_name, "shards": len(shards), "start": start, "dry_run": True}

    log.info("resolving shards (HF cache hits are free, 0 API calls)…")
    local_shards = download_many(repo, shards, token)

    cfg_dir = getattr(cfg.paths, f"{config_name}_dir")
    work = os.path.join(cfg_dir, "compact")
    made = bunch_local_parquets(
        local_shards, work, start_index=start,
        bunch_size=bunch_size, bunch_max_bytes=max_bytes,
    )
    prefix = data_prefix(config_name)
    api = HfApi(token=token)

    add_ops = [
        CommitOperationAdd(
            path_in_repo=f"{prefix}{BUNCH_PREFIX}{b['index']:05d}.parquet",
            path_or_fileobj=b["path"],
        )
        for b in made
    ]
    log.info("uploading %d bunch file(s) — training can start as soon as this commit lands",
             len(add_ops))
    _commit_ops(api, repo, add_ops,
                f"{config_name}: compact {len(shards)} shards -> {len(made)} bunches")
    log.info("bunches are on Hub. you can start training now:\n"
             "  python scripts/03_train.py --from-hub")

    _update_hub_state_after_compact(cfg, config_name, made, api, repo)

    del_ops = [CommitOperationDelete(path_in_repo=p) for p in shards]
    batch = 200
    for i in range(0, len(del_ops), batch):
        chunk = del_ops[i:i + batch]
        _commit_ops(
            api, repo, chunk,
            f"{config_name}: delete tiny shards {i + 1}-{i + len(chunk)}/{len(del_ops)}",
        )
    log.info("deleted %d tiny %s shards from Hub", len(shards), config_name)

    if write_local:
        local_files = download_many(repo, bunches, token) if bunches else []
        local_files.extend(b["path"] for b in made)
        write_local_dataset(cfg, config_name, local_files)

    for b in made:
        try:
            os.remove(b["path"])
        except OSError:
            pass
    return {"config": config_name, "compacted": len(shards), "bunches": len(made)}


def write_local_dataset(cfg, config_name: str, parquet_paths: list[str]) -> str:
    """Materialize a `save_to_disk` dataset so train skips the Hub entirely."""
    from datasets import load_dataset

    dest = os.path.join(getattr(cfg.paths, f"{config_name}_dir"), "dataset")
    log.info("writing local %s dataset (%d parquet) -> %s", config_name, len(parquet_paths), dest)
    ds = load_dataset("parquet", data_files={"train": parquet_paths}, split="train")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    ds.save_to_disk(dest)
    log.info("local %s ready: %d rows at %s  (train will use this, no Hub)",
             config_name, ds.num_rows, dest)
    return dest


def _update_hub_state_after_compact(cfg, config_name: str, made: list[dict],
                                    api, repo: str) -> None:
    """Rewrite resume JSON so encode/fetch do not re-process compacted files."""
    from huggingface_hub import CommitOperationAdd

    from .fetch_state import load_json, save_json

    token = require_token()
    ops = []

    if config_name == "mimi":
        from huggingface_hub.utils import EntryNotFoundError

        local = os.path.join(cfg.paths.mimi_dir, "encode_state.json")
        st = load_json(local) or {}
        hub = {}
        try:
            from huggingface_hub import hf_hub_download
            hub_p = hf_hub_download(repo, "mimi/encode_state.json",
                                    repo_type="dataset", token=token)
            hub = load_json(hub_p) or {}
        except (EntryNotFoundError, OSError, Exception):
            hub = {}
        done = sorted(set(st.get("audio_files_done", [])) | set(hub.get("audio_files_done", [])))
        nxt_bunch = max(
            int(st.get("next_bunch_index") or 0),
            int(hub.get("next_bunch_index") or 0),
            next_bunch_index([f"bunch_{b['index']:05d}.parquet" for b in made]),
        )
        snap = {
            "audio_files_done": done,
            "next_mimi_index": 0,
            "next_bunch_index": nxt_bunch,
            "bunched": True,
            "bunches": [
                {
                    "index": b["index"],
                    "rows": b["rows"],
                    "hub_path": f"mimi/data/bunch_{b['index']:05d}.parquet",
                    "source_shards": b["sources"],
                }
                for b in made
            ],
        }
        os.makedirs(cfg.paths.mimi_dir, exist_ok=True)
        save_json(local, snap)
        ops.append(CommitOperationAdd("mimi/encode_state.json", local))

    elif config_name == "audio":
        from .fetch_state import (
            HUB_FPS, HUB_STATE, load_merged_state, local_paths, persist_state, upsert_shard,
        )

        state, seen, _ = load_merged_state(cfg)
        state["bunched"] = True
        state["next_bunch_index"] = max(
            int(state.get("next_bunch_index") or 0),
            next_bunch_index([f"bunch_{b['index']:05d}.parquet" for b in made]),
        )
        state.setdefault("bunches", [])
        for b in made:
            state["bunches"].append({
                "index": b["index"],
                "rows": b["rows"],
                "hub_path": f"audio/data/bunch_{b['index']:05d}.parquet",
                "uploaded": True,
            })
            for src in b.get("sources") or []:
                try:
                    idx = int(src[len("shard_"):-len(".parquet")])
                    upsert_shard(state, idx, rows=0, uploaded=True)
                except ValueError:
                    pass
        persist_state(cfg, state, seen)
        ls, lf = local_paths(cfg.paths.audio_dir)
        ops.append(CommitOperationAdd(HUB_STATE, ls))
        if os.path.isfile(lf):
            ops.append(CommitOperationAdd(HUB_FPS, lf))

    if ops:
        _commit_ops(api, repo, ops, f"{config_name}: update resume state after compact")
        log.info("updated %s resume JSON on Hub", config_name)
