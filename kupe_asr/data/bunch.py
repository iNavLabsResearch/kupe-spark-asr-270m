"""Pack many tiny parquet shards into a few Hub files.

Free-tier Hub allows 1000 API requests / 5 minutes. Each parquet download is
at least one request, so ~1000 shard_*.parquet files 429 before training
starts. Bunching 1000 shards -> 1 file drops mimi from ~1000 requests to 1.

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
DEFAULT_BUNCH_SIZE = 1000      # shards per Hub file (one Hub quota window)
DEFAULT_BUNCH_MAX_MB = 1500    # audio cap so a bunch stays ~1.5 GB, not 150 GB
HF_QUOTA_WINDOW_S = 300        # free-tier Hub: 1000 API requests / 5 minutes
HF_QUOTA_SOFT = 900            # sleep before we actually hit 1000


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
    """hf_hub_download with cache-first. On 429, sleep 5 min for the Hub quota reset."""
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    hit = _try_cache(repo_id, path_in_repo, token)
    if hit:
        return hit
    while True:
        try:
            return hf_hub_download(
                repo_id, path_in_repo, repo_type="dataset", token=token,
            )
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            log.warning("Hub 429 on %s — sleeping 5 min for quota reset, then retry",
                        path_in_repo)
            time.sleep(HF_QUOTA_WINDOW_S)


def _try_cache(repo_id: str, path_in_repo: str, token: str) -> str | None:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        return hf_hub_download(
            repo_id, path_in_repo, repo_type="dataset", token=token, local_files_only=True,
        )
    except Exception:
        return None


def download_many(repo_id: str, paths: list[str], token: str) -> list[str]:
    """Resolve Hub parquet paths. Cache hits are free; every 900 network downloads
    we sleep 5 min so we never trip the 1000-req quota."""
    local: list[str] = []
    net = 0
    for i, p in enumerate(paths, 1):
        hit = _try_cache(repo_id, p, token)
        if hit:
            local.append(hit)
        else:
            if net > 0 and net % HF_QUOTA_SOFT == 0:
                log.info("downloaded %d files this window — sleeping 5 min to reset Hub quota…",
                         net)
                time.sleep(HF_QUOTA_WINDOW_S)
            local.append(download_hub_file(repo_id, p, token))
            net += 1
        if i == 1 or i % 50 == 0 or i == len(paths):
            log.info("resolved %d/%d (%d cache, %d network)  (%s)",
                     i, len(paths), i - net, net, p)
    return local


LEDGER_NAME = "compact_ledger.json"


def ledger_local_path(cfg, config_name: str) -> str:
    return os.path.join(getattr(cfg.paths, f"{config_name}_dir"), LEDGER_NAME)


def ledger_hub_path(config_name: str) -> str:
    return f"{config_name}/{LEDGER_NAME}"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_ledger(repo: str, config_name: str, bunch_size: int) -> dict:
    return {
        "version": 1,
        "repo": repo,
        "config": config_name,
        "bunch_size": bunch_size,
        "status": "in_progress",
        "updated_at": _now(),
        "plan": [],
        "bunches": [],
    }


def _bunch_entry(ledger: dict, idx: int) -> dict | None:
    for b in ledger.get("bunches") or []:
        if int(b["index"]) == idx:
            return b
    return None


def _upsert_bunch(ledger: dict, entry: dict) -> None:
    bunches = ledger.setdefault("bunches", [])
    for i, b in enumerate(bunches):
        if int(b["index"]) == int(entry["index"]):
            bunches[i] = {**b, **entry}
            return
    bunches.append(entry)
    bunches.sort(key=lambda b: int(b["index"]))


def save_ledger(path: str, ledger: dict) -> None:
    """Atomic JSON write so a crash never leaves a half-written ledger."""
    from .fetch_state import save_json
    ledger = dict(ledger)
    ledger["updated_at"] = _now()
    save_json(path, ledger)
    log.info("ledger saved %s  status=%s  bunches=%d",
             path, ledger.get("status"), len(ledger.get("bunches") or []))


def load_ledger(cfg, config_name: str, repo: str, bunch_size: int, token: str) -> dict:
    """Local ledger wins; Hub copy fills gaps so a new box can resume too."""
    from .fetch_state import load_json

    local_p = ledger_local_path(cfg, config_name)
    local = load_json(local_p) or {}
    hub = {}
    try:
        from huggingface_hub import hf_hub_download
        hp = hf_hub_download(repo, ledger_hub_path(config_name),
                             repo_type="dataset", token=token)
        hub = load_json(hp) or {}
    except Exception:
        hub = {}

    if not local and not hub:
        return _empty_ledger(repo, config_name, bunch_size)

    out = _empty_ledger(repo, config_name, bunch_size)
    out["bunch_size"] = int(local.get("bunch_size") or hub.get("bunch_size") or bunch_size)
    plan_a, plan_b = local.get("plan") or [], hub.get("plan") or []
    out["plan"] = plan_a if len(plan_a) >= len(plan_b) else plan_b
    by_idx: dict[int, dict] = {}
    for src in (hub.get("bunches") or [], local.get("bunches") or []):
        for b in src:
            idx = int(b["index"])
            prev = by_idx.get(idx, {})
            merged = {**prev, **b}
            for key in ("concatenated_at", "uploaded_at", "sources_deleted_at"):
                merged[key] = b.get(key) or prev.get(key)
            by_idx[idx] = merged
    out["bunches"] = [by_idx[i] for i in sorted(by_idx)]
    if local.get("status") == "done" or hub.get("status") == "done":
        out["status"] = "done"
    save_ledger(local_p, out)
    return out


def compact_status(cfg, config_name: str = "mimi") -> dict:
    from .fetch_state import load_json
    token = require_token()
    repo = cfg.repos.data
    bunches, shards = list_config_parquets(repo, config_name, token)
    ledger = load_json(ledger_local_path(cfg, config_name)) or {}
    uploaded = [b for b in (ledger.get("bunches") or []) if b.get("uploaded_at")]
    pending = [p for p in (ledger.get("plan") or [])
               if not (_bunch_entry(ledger, int(p["index"])) or {}).get("uploaded_at")]
    log.info("===== compact ledger (%s) =====", config_name)
    log.info("  Hub: %d bunch(es), %d leftover tiny shard(s)", len(bunches), len(shards))
    log.info("  ledger status: %s", ledger.get("status") or "none")
    log.info("  bunches uploaded: %d / %d planned",
             len(uploaded), len(ledger.get("plan") or []))
    if pending:
        log.info("  still to do: bunch indices %s", [int(p["index"]) for p in pending])
    log.info("  re-run the same command to resume. originals stay on Hub until uploaded.")
    log.info("================================")
    return ledger


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
    """Pack `{config}/data/shard_*.parquet` into `bunch_*.parquet`.

    Safe: originals on Hub are never deleted until that bunch is uploaded AND
    the JSON ledger records `uploaded_at`. Crash → re-run; ledger resumes.

    Ledger: `{audio|mimi}_dir/compact_ledger.json` (also pushed to Hub each bunch).
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    token = require_token()
    repo = cfg.repos.data
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    prefix = data_prefix(config_name)
    cfg_dir = getattr(cfg.paths, f"{config_name}_dir")
    work = os.path.join(cfg_dir, "compact")
    ledger_p = ledger_local_path(cfg, config_name)
    os.makedirs(cfg_dir, exist_ok=True)

    bunches, shards = list_config_parquets(repo, config_name, token)
    log.info("%s Hub: %d bunch(es), %d shard(s)", config_name, len(bunches), len(shards))
    ledger = load_ledger(cfg, config_name, repo, bunch_size, token)
    done_sources = set()
    for b in ledger.get("bunches") or []:
        if b.get("uploaded_at"):
            done_sources.update(b.get("source_shards") or [])

    leftover = [s for s in shards if s not in done_sources]
    need_delete = any(
        b.get("uploaded_at") and not b.get("sources_deleted_at")
        for b in (ledger.get("bunches") or [])
    )

    if not leftover and not shards and not need_delete:
        log.info("nothing to compact for `%s` — already bunched or empty", config_name)
        if write_local and bunches:
            files = download_many(repo, bunches, token)
            write_local_dataset(cfg, config_name, files)
        ledger["status"] = "done"
        save_ledger(ledger_p, ledger)
        return {"config": config_name, "bunches": len(bunches), "compacted": 0}

    if not leftover and not need_delete and ledger.get("plan") and all(
        (_bunch_entry(ledger, int(p["index"])) or {}).get("uploaded_at")
        for p in ledger["plan"]
    ):
        log.info("`%s` ledger already complete", config_name)
        if write_local:
            hub_bunches, _ = list_config_parquets(repo, config_name, token)
            files = [b.get("local_path") for b in (ledger.get("bunches") or [])
                     if b.get("local_path") and os.path.isfile(b["local_path"])]
            files = files or download_many(repo, hub_bunches, token)
            if files:
                write_local_dataset(cfg, config_name, files)
        ledger["status"] = "done"
        save_ledger(ledger_p, ledger)
        return {"config": config_name, "bunches": len(ledger.get("bunches") or []), "compacted": 0}

    if not ledger.get("plan"):
        start = next_bunch_index(bunches)
        if leftover:
            groups = group_paths(leftover, bunch_size=bunch_size,
                                 bunch_max_bytes=int(bunch_max_mb) * 1024 * 1024 if bunch_max_mb else 0)
            ledger["plan"] = [
                {"index": start + i, "source_shards": g} for i, g in enumerate(groups)
            ]
        else:
            ledger["plan"] = [
                {"index": int(b["index"]), "source_shards": list(b.get("source_shards") or [])}
                for b in (ledger.get("bunches") or [])
            ]
        save_ledger(ledger_p, ledger)
        log.info("ledger plan: %d bunch(es) from %d leftover shards",
                 len(ledger["plan"]), len(leftover))

    if dry_run:
        return {"config": config_name, "plan": ledger["plan"], "dry_run": True}

    api = HfApi(token=token)
    made_local: list[dict] = []

    for step in ledger["plan"]:
        idx = int(step["index"])
        sources = list(step["source_shards"])
        existing = _bunch_entry(ledger, idx) or {}
        hub_path = f"{prefix}{BUNCH_PREFIX}{idx:05d}.parquet"
        local_bunch = existing.get("local_path") or os.path.join(
            work, f"{BUNCH_PREFIX}{idx:05d}.parquet"
        )

        if existing.get("uploaded_at"):
            log.info("bunch_%05d already uploaded (%s) — skip", idx, existing["uploaded_at"])
            if not existing.get("sources_deleted_at"):
                _, still = list_config_parquets(repo, config_name, token)
                to_del = [s for s in sources if s in still]
                del_ops = [CommitOperationDelete(path_in_repo=s) for s in to_del]
                for i in range(0, len(del_ops), 200):
                    chunk = del_ops[i:i + 200]
                    _commit_ops(
                        api, repo, chunk,
                        f"{config_name}: delete shards for bunch_{idx:05d} "
                        f"({i + 1}-{i + len(chunk)}/{len(del_ops)})",
                    )
                _upsert_bunch(ledger, {**existing, "sources_deleted_at": _now()})
                save_ledger(ledger_p, ledger)
            if os.path.isfile(local_bunch):
                made_local.append({
                    "index": idx, "path": local_bunch,
                    "rows": existing.get("rows") or 0,
                    "sources": [os.path.basename(s) for s in sources],
                })
            continue

        log.info("=== bunch_%05d: resolving %d shards ===", idx, len(sources))
        local_shards = download_many(repo, sources, token)

        if not (existing.get("concatenated_at") and os.path.isfile(local_bunch)):
            os.makedirs(work, exist_ok=True)
            rows = concat_parquets(local_shards, local_bunch)
            _upsert_bunch(ledger, {
                "index": idx,
                "hub_path": hub_path,
                "local_path": local_bunch,
                "rows": rows,
                "bytes": os.path.getsize(local_bunch),
                "source_shards": sources,
                "concatenated_at": _now(),
                "uploaded_at": None,
                "sources_deleted_at": None,
            })
            save_ledger(ledger_p, ledger)
            log.info("bunch_%05d concatenated %d shards -> %d rows (%.1f MB)  [ledger]",
                     idx, len(sources), rows, os.path.getsize(local_bunch) / 1e6)
        else:
            rows = int(existing.get("rows") or 0)
            log.info("bunch_%05d concat already on disk — resume upload", idx)

        log.info("uploading %s …", hub_path)
        ops = [
            CommitOperationAdd(path_in_repo=hub_path, path_or_fileobj=local_bunch),
            CommitOperationAdd(path_in_repo=ledger_hub_path(config_name),
                               path_or_fileobj=ledger_p),
        ]
        _commit_ops(api, repo, ops,
                    f"{config_name}: bunch_{idx:05d} ({len(sources)} shards packed)")
        existing = _bunch_entry(ledger, idx) or {}
        _upsert_bunch(ledger, {**existing, "uploaded_at": _now()})
        save_ledger(ledger_p, ledger)
        _commit_ops(
            api, repo,
            [CommitOperationAdd(path_in_repo=ledger_hub_path(config_name),
                                path_or_fileobj=ledger_p)],
            f"{config_name}: ledger uploaded bunch_{idx:05d}",
        )
        log.info("bunch_%05d ON HUB — originals still present until delete step", idx)

        _, still = list_config_parquets(repo, config_name, token)
        to_del = [s for s in sources if s in still]
        del_ops = [CommitOperationDelete(path_in_repo=s) for s in to_del]
        for i in range(0, len(del_ops), 200):
            chunk = del_ops[i:i + 200]
            _commit_ops(
                api, repo, chunk,
                f"{config_name}: delete shards for bunch_{idx:05d} "
                f"({i + 1}-{i + len(chunk)}/{len(del_ops)})",
            )
        existing = _bunch_entry(ledger, idx) or {}
        _upsert_bunch(ledger, {**existing, "sources_deleted_at": _now()})
        save_ledger(ledger_p, ledger)
        log.info("bunch_%05d source shards deleted from Hub  [ledger]", idx)

        made_local.append({
            "index": idx, "path": local_bunch, "rows": rows,
            "sources": [os.path.basename(s) for s in sources],
        })

    if made_local:
        _update_hub_state_after_compact(cfg, config_name, made_local, api, repo)

    if write_local:
        local_files = [b["path"] for b in made_local if os.path.isfile(b["path"])]
        if not local_files:
            hub_bunches, _ = list_config_parquets(repo, config_name, token)
            local_files = download_many(repo, hub_bunches, token)
        if local_files:
            write_local_dataset(cfg, config_name, local_files)

    ledger["status"] = "done"
    save_ledger(ledger_p, ledger)
    _commit_ops(
        api, repo,
        [CommitOperationAdd(path_in_repo=ledger_hub_path(config_name),
                            path_or_fileobj=ledger_p)],
        f"{config_name}: compact ledger done",
    )
    log.info("compact DONE for %s — originals were only removed after each bunch landed",
             config_name)
    return {
        "config": config_name,
        "compacted": sum(len(p["source_shards"]) for p in ledger["plan"]),
        "bunches": len(ledger["plan"]),
    }


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
        by_b: dict[int, dict] = {}
        for src in (st.get("bunches") or [], hub.get("bunches") or []):
            for b in src:
                by_b[int(b["index"])] = dict(b)
        for b in made:
            by_b[int(b["index"])] = {
                "index": b["index"],
                "rows": b["rows"],
                "hub_path": f"mimi/data/bunch_{b['index']:05d}.parquet",
                "source_shards": b["sources"],
            }
        nxt_bunch = max(
            int(st.get("next_bunch_index") or 0),
            int(hub.get("next_bunch_index") or 0),
            (max(by_b) + 1) if by_b else 0,
        )
        snap = {
            "audio_files_done": done,
            "next_mimi_index": 0,
            "next_bunch_index": nxt_bunch,
            "bunched": True,
            "bunches": [by_b[i] for i in sorted(by_b)],
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
