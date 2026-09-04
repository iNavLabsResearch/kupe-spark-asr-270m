"""Resume JSON + per-shard Hub uploads for fetch.

Local:  artifacts/data/audio/fetch_state.json  + seen_fps.txt
Hub:    audio/fetch_state.json                 + audio/seen_fps.txt
        audio/data/shard_XXXXX.parquet
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

from ..constants import LANGUAGES
from ..hf_utils import log, require_token, upload_file


STATE_NAME = "fetch_state.json"
FPS_NAME = "seen_fps.txt"
HUB_STATE = f"audio/{STATE_NAME}"
HUB_FPS = f"audio/{FPS_NAME}"
HUB_SHARD_DIR = "audio/data"


def clip_fp(lang: str, source: str, text: str, duration: float) -> str:
    raw = f"{lang}|{source}|{text}|{float(duration):.2f}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def empty_lang() -> dict:
    return {
        "status": "pending",          # pending | in_progress | done
        "kept_seconds": 0.0,
        "val_seconds": 0.0,
        "n_clips": 0,
        "by_source": {},
        "next_clip_index": 0,
        "files_done": {},             # source -> parquet files fully consumed (resume point)
    }


def empty_state(repo_id: str, languages: list[str]) -> dict:
    return {
        "version": 1,
        "repo": repo_id,
        "config": "audio",
        "updated_at": None,
        "next_shard_index": 0,
        "next_bunch_index": 0,
        "bunched": False,
        "bunches": [],            # [{index, rows, uploaded, hub_path}]
        "shards": [],             # [{index, rows, uploaded, hub_path}]
        "languages": {lang: empty_lang() for lang in languages},
    }


def local_paths(audio_dir: str) -> tuple[str, str]:
    os.makedirs(audio_dir, exist_ok=True)
    return (
        os.path.join(audio_dir, STATE_NAME),
        os.path.join(audio_dir, FPS_NAME),
    )


def load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_fps(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h = line.strip()
            if h:
                out.add(h)
    return out


def save_fps(path: str, fps: set[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for h in sorted(fps):
            f.write(h + "\n")
    os.replace(tmp, path)


def _download_hub(repo_id: str, path_in_repo: str, token: str) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        return hf_hub_download(
            repo_id, path_in_repo, repo_type="dataset", token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        return None
    except Exception as e:
        log.warning("hub download %s failed: %s", path_in_repo, e)
        return None


def list_hub_shard_indices(repo_id: str, token: str) -> set[int]:
    from huggingface_hub import HfApi

    found: set[int] = set()
    try:
        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        log.warning("could not list Hub files: %s", e)
        return found
    prefix = f"{HUB_SHARD_DIR}/shard_"
    for name in files:
        if not (name.startswith(prefix) and name.endswith(".parquet")):
            continue
        stem = os.path.basename(name)[len("shard_"):-len(".parquet")]
        try:
            found.add(int(stem))
        except ValueError:
            continue
    return found


def list_local_shards(shards_dir: str) -> list[tuple[int, str]]:
    """Return sorted (index, arrow-dir) for shard_XXXXX folders."""
    if not os.path.isdir(shards_dir):
        return []
    out: list[tuple[int, str]] = []
    for name in os.listdir(shards_dir):
        path = os.path.join(shards_dir, name)
        if not (os.path.isdir(path) and name.startswith("shard_")):
            continue
        if not os.path.isfile(os.path.join(path, "dataset_info.json")):
            continue
        try:
            idx = int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((idx, path))
    out.sort()
    return out


def _shard_entry(idx: int, rows: int, uploaded: bool) -> dict:
    return {
        "index": idx,
        "rows": rows,
        "uploaded": uploaded,
        "hub_path": f"{HUB_SHARD_DIR}/shard_{idx:05d}.parquet",
    }


def upsert_shard(state: dict, idx: int, rows: int, uploaded: bool) -> None:
    shards = state.setdefault("shards", [])
    for s in shards:
        if int(s["index"]) == idx:
            s["rows"] = max(int(s.get("rows") or 0), int(rows or 0))
            s["uploaded"] = bool(uploaded) or bool(s.get("uploaded"))
            s["hub_path"] = f"{HUB_SHARD_DIR}/shard_{idx:05d}.parquet"
            break
    else:
        shards.append(_shard_entry(idx, rows, uploaded))
    shards.sort(key=lambda s: int(s["index"]))
    nxt = max((int(s["index"]) for s in shards), default=-1) + 1
    state["next_shard_index"] = max(int(state.get("next_shard_index", 0)), nxt)


def merge_states(a: dict, b: dict, languages: list[str]) -> dict:
    """Prefer whichever side has more kept seconds / more shards uploaded."""
    if not a:
        return b
    if not b:
        return a
    out = empty_state(a.get("repo") or b.get("repo") or "", languages)
    for lang in languages:
        la = a.get("languages", {}).get(lang, empty_lang())
        lb = b.get("languages", {}).get(lang, empty_lang())
        pick = la if float(la.get("kept_seconds", 0)) >= float(lb.get("kept_seconds", 0)) else lb
        merged = empty_lang()
        merged.update(pick)
        # union by_source with max counts
        by = dict(la.get("by_source") or {})
        for k, v in (lb.get("by_source") or {}).items():
            by[k] = max(int(by.get(k, 0)), int(v))
        merged["by_source"] = by
        merged["n_clips"] = max(int(la.get("n_clips", 0)), int(lb.get("n_clips", 0)))
        merged["next_clip_index"] = max(
            int(la.get("next_clip_index", 0)), int(lb.get("next_clip_index", 0)),
            merged["n_clips"],
        )
        merged["val_seconds"] = max(float(la.get("val_seconds", 0)), float(lb.get("val_seconds", 0)))
        merged["kept_seconds"] = max(float(la.get("kept_seconds", 0)), float(lb.get("kept_seconds", 0)))
        if merged["status"] == "pending" and (la.get("status") == "done" or lb.get("status") == "done"):
            merged["status"] = "done"
        elif merged["kept_seconds"] > 0 and merged["status"] == "pending":
            merged["status"] = "in_progress"
        out["languages"][lang] = merged

    by_idx: dict[int, dict] = {}
    for src in (a, b):
        for s in src.get("shards") or []:
            idx = int(s["index"])
            prev = by_idx.get(idx)
            if prev is None:
                by_idx[idx] = dict(s)
            else:
                prev["uploaded"] = bool(prev.get("uploaded")) or bool(s.get("uploaded"))
                prev["rows"] = max(int(prev.get("rows") or 0), int(s.get("rows") or 0))
    for idx in sorted(by_idx):
        out["shards"].append(by_idx[idx])
    out["next_shard_index"] = max(
        int(a.get("next_shard_index") or 0),
        int(b.get("next_shard_index") or 0),
        (max(by_idx) + 1) if by_idx else 0,
    )
    out["next_bunch_index"] = max(
        int(a.get("next_bunch_index") or 0),
        int(b.get("next_bunch_index") or 0),
    )
    out["bunched"] = bool(a.get("bunched") or b.get("bunched"))
    by_bunch: dict[int, dict] = {}
    for src in (a, b):
        for bun in src.get("bunches") or []:
            by_bunch[int(bun["index"])] = dict(bun)
    out["bunches"] = [by_bunch[i] for i in sorted(by_bunch)]
    out["repo"] = a.get("repo") or b.get("repo")
    return out


def ingest_local_shard(state: dict, languages: list[str], idx: int, arrow_dir: str,
                       seen: set[str]) -> int:
    """Read metadata from a local arrow shard into state + seen fingerprints."""
    from datasets import load_from_disk

    ds = load_from_disk(arrow_dir)
    drop = [c for c in ds.column_names if c == "audio"]
    meta = ds.remove_columns(drop) if drop else ds
    n = 0
    for row in meta:
        lang = row["language"]
        src = row["source"]
        text = row["text"]
        dur = float(row["duration"])
        seen.add(clip_fp(lang, src, text, dur))
        st = state["languages"].setdefault(lang, empty_lang())
        st["kept_seconds"] = float(st.get("kept_seconds", 0)) + dur
        st["n_clips"] = int(st.get("n_clips", 0)) + 1
        try:
            n_id = int(str(row.get("id", "")).rsplit("-", 1)[-1])
            st["next_clip_index"] = max(int(st.get("next_clip_index", 0)), n_id + 1)
        except ValueError:
            st["next_clip_index"] = max(int(st.get("next_clip_index", 0)), int(st["n_clips"]))
        by = st.setdefault("by_source", {})
        by[src] = int(by.get(src, 0)) + 1
        if row.get("split") == "val":
            st["val_seconds"] = float(st.get("val_seconds", 0)) + dur
        if st["status"] == "pending":
            st["status"] = "in_progress"
        n += 1
    upsert_shard(state, idx, n, uploaded=False)
    log.info("indexed local shard_%05d (%d rows)", idx, n)
    return n


def format_progress(state: dict, caps) -> str:
    parts = []
    for lang in LANGUAGES:
        st = state.get("languages", {}).get(lang) or empty_lang()
        cap_h = float(getattr(caps, lang, 0) or 0)
        got_h = float(st.get("kept_seconds", 0)) / 3600.0
        parts.append(f"{lang} {got_h:.1f}/{cap_h:.0f}h")
    n_up = sum(1 for s in state.get("shards") or [] if s.get("uploaded"))
    n_all = len(state.get("shards") or [])
    return f"shards {n_up}/{n_all} uploaded | " + " ".join(parts)


def print_status(state: dict, caps, hub_indices: set[int]) -> None:
    log.info("===== fetch resume status =====")
    log.info("%s", format_progress(state, caps))
    for lang, name in LANGUAGES.items():
        st = state.get("languages", {}).get(lang) or empty_lang()
        cap_s = float(getattr(caps, lang, 0) or 0) * 3600.0
        got = float(st.get("kept_seconds", 0))
        log.info(
            "  %s (%s): %s  %.1f h  %d clips  val %.1f min  sources=%s",
            lang, name, st.get("status"), got / 3600, int(st.get("n_clips", 0)),
            float(st.get("val_seconds", 0)) / 60, st.get("by_source") or {},
        )
        if cap_s > 0 and got >= cap_s:
            log.info("           already at cap — will skip")
    local_pending = [
        s for s in (state.get("shards") or [])
        if not s.get("uploaded") and int(s["index"]) not in hub_indices
    ]
    on_hub = sorted(hub_indices)
    n_bunch = len(state.get("bunches") or [])
    log.info("  Hub parquet shards: %s", on_hub[:8] + (["…", on_hub[-1]] if len(on_hub) > 8 else on_hub[8:]))
    if n_bunch or state.get("bunched"):
        log.info("  Hub bunches: %d (tiny shards packed 200-to-1)", n_bunch)
    log.info("  local shards still to upload: %s", [int(s["index"]) for s in local_pending] or "none")
    log.info("================================")


def audio_data_card(project: str, langs: str, data_repo: str) -> str:
    return f"""\
---
pretty_name: {project}-data
tags:
  - asr
  - audio
  - speech
configs:
  - config_name: audio
    data_files:
      - split: train
        path: audio/data/*.parquet
  - config_name: mimi
    data_files:
      - split: train
        path: mimi/data/*.parquet
---

# {project} — data

Multilingual ASR corpus for **{project}** (Gemma-3-270m + Mimi codec).

Languages: {langs}

## Configs
- **audio** — raw speech resampled to 24 kHz mono (`audio/data/bunch_*.parquet`).
- **mimi** — Mimi codebook-0 tokens (12.5 tok/s) + transcripts (`mimi/data/bunch_*.parquet`). Used for training.

Tiny `shard_*.parquet` files are packed 200-to-1 into `bunch_*.parquet` so Hub
downloads stay under the free-tier 1000-request / 5-minute cap. Resume state
lives in `audio/fetch_state.json`.

```python
from datasets import load_dataset
audio = load_dataset("{data_repo}", "audio", split="train")
mimi  = load_dataset("{data_repo}", "mimi",  split="train")
```

Sources: ai4bharat/IndicVoices, ARTPARK-IISc/Vaani-transcription-part,
mozilla-foundation/common_voice_17_0, MLCommons/peoples_speech.
See each source for its license.
"""


def ensure_data_card(cfg) -> None:
    langs = ", ".join(f"{c} ({n})" for c, n in LANGUAGES.items())
    card = audio_data_card(cfg.project, langs, cfg.repos.data)
    token = require_token()
    existing = _download_hub(cfg.repos.data, "README.md", token)
    if existing:
        with open(existing, "r", encoding="utf-8") as f:
            text = f.read()
        if "audio/data/*.parquet" in text and "mimi/data/*.parquet" in text:
            return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "README.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(card)
        upload_file(p, cfg.repos.data, "dataset", "README.md",
                    "dataset card: audio parquet glob")
    log.info("Hub README updated with audio/data/*.parquet config")


def load_merged_state(cfg, *, reset: bool = False) -> tuple[dict, set[str], set[int]]:
    """Local JSON + Hub JSON + Hub parquet listing. reset=True ignores both."""
    token = require_token()
    langs = list(cfg.languages)
    local_state_path, local_fps_path = local_paths(cfg.paths.audio_dir)
    hub_indices = list_hub_shard_indices(cfg.repos.data, token)

    if reset:
        log.warning("--reset: ignoring local + Hub fetch_state (Hub parquets still listed)")
        state = empty_state(cfg.repos.data, langs)
        for idx in hub_indices:
            upsert_shard(state, idx, rows=0, uploaded=True)
        return state, set(), hub_indices

    local_st = load_json(local_state_path)
    hub_path = _download_hub(cfg.repos.data, HUB_STATE, token)
    hub_st = load_json(hub_path) if hub_path else None
    state = merge_states(
        local_st or empty_state(cfg.repos.data, langs),
        hub_st or empty_state(cfg.repos.data, langs),
        langs,
    )
    state["repo"] = cfg.repos.data
    for lang in langs:
        state["languages"].setdefault(lang, empty_lang())
    for idx in hub_indices:
        upsert_shard(state, idx, rows=0, uploaded=True)

    seen = load_fps(local_fps_path)
    hub_fps = _download_hub(cfg.repos.data, HUB_FPS, token)
    if hub_fps:
        seen |= load_fps(hub_fps)
    return state, seen, hub_indices


def persist_state(cfg, state: dict, seen: set[str]) -> tuple[str, str]:
    state = dict(state)
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    local_state_path, local_fps_path = local_paths(cfg.paths.audio_dir)
    save_json(local_state_path, state)
    save_fps(local_fps_path, seen)
    return local_state_path, local_fps_path


def commit_shard(cfg, state: dict, seen: set[str], parquet_path: str, idx: int, rows: int) -> None:
    """One Hub commit: parquet + fetch_state.json + seen_fps.txt."""
    from huggingface_hub import CommitOperationAdd, HfApi

    persist_state(cfg, state, seen)
    local_state_path, local_fps_path = local_paths(cfg.paths.audio_dir)
    remote = f"{HUB_SHARD_DIR}/shard_{idx:05d}.parquet"
    api = HfApi(token=require_token())
    ops = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=parquet_path),
        CommitOperationAdd(path_in_repo=HUB_STATE, path_or_fileobj=local_state_path),
        CommitOperationAdd(path_in_repo=HUB_FPS, path_or_fileobj=local_fps_path),
    ]
    api.create_commit(
        repo_id=cfg.repos.data,
        repo_type="dataset",
        operations=ops,
        commit_message=f"audio shard_{idx:05d} ({rows} rows)",
    )
    upsert_shard(state, idx, rows, uploaded=True)
    persist_state(cfg, state, seen)
    log.info(
        "HUB UPLOAD ok  shard_%05d  %d rows  ->  %s/%s",
        idx, rows, cfg.repos.data, remote,
    )
    log.info("resume  %s", format_progress(state, cfg.data.max_hours_per_lang))


def arrow_to_parquet(arrow_dir: str, parquet_path: str) -> int:
    from datasets import load_from_disk

    ds = load_from_disk(arrow_dir)
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    ds.to_parquet(parquet_path)
    return ds.num_rows


def pending_dir(cfg) -> str:
    d = os.path.join(cfg.paths.audio_dir, "pending")
    os.makedirs(d, exist_ok=True)
    return d


def stage_to_pending(cfg, arrow_dir: str, idx: int, *, drop_local: bool) -> tuple[str, int]:
    """Convert a shard to parquet and move it to the local pending/ dir (no Hub commit)."""
    pq_path, rows = stage_shard(cfg, arrow_dir, idx, drop_local=drop_local)
    dest = os.path.join(pending_dir(cfg), f"shard_{idx:05d}.parquet")
    shutil.move(pq_path, dest)
    return dest, rows


def _commit_with_backoff(fn, what: str) -> None:
    delays = [0, 30, 120, 300, 900, 1800]
    for attempt, delay in enumerate(delays):
        if delay:
            log.warning("%s backoff %ds (attempt %d/%d)…", what, delay, attempt + 1, len(delays))
            time.sleep(delay)
        try:
            fn()
            return
        except Exception as e:
            rate = "429" in str(e) or "Too Many Requests" in str(e)
            log.warning("%s failed%s: %s", what, " (rate-limited)" if rate else "", e)
            if not rate or attempt == len(delays) - 1:   # only back off on rate limits
                raise


def upload_pending(cfg, state: dict, seen: set[str]) -> int:
    """Pack pending tiny shards into bunch_*.parquet and upload those.

    `data.bunch_size` (default 200) is the max shards per Hub file.
    `data.bunch_max_mb` (default 1500) also caps audio bunches so a file
    stays ~1.5 GB rather than 200 × 150 MB = 30 GB.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    from .bunch import bunch_local_parquets, list_config_parquets, next_bunch_index as _nbi

    pdir = pending_dir(cfg)
    files = sorted(f for f in os.listdir(pdir) if f.endswith(".parquet"))
    api = HfApi(token=require_token())

    if files:
        bunch_size = int(getattr(cfg.data, "bunch_size", 200))
        bunch_max_mb = int(getattr(cfg.data, "bunch_max_mb", 1500))
        hub_bunches, _ = list_config_parquets(cfg.repos.data, "audio", require_token())
        start = max(int(state.get("next_bunch_index") or 0), _nbi(hub_bunches))
        out_dir = os.path.join(cfg.paths.audio_dir, "bunches")
        made = bunch_local_parquets(
            [os.path.join(pdir, f) for f in files],
            out_dir, start_index=start,
            bunch_size=bunch_size,
            bunch_max_bytes=bunch_max_mb * 1024 * 1024 if bunch_max_mb else 0,
        )
        ops = [
            CommitOperationAdd(
                path_in_repo=f"{HUB_SHARD_DIR}/bunch_{b['index']:05d}.parquet",
                path_or_fileobj=b["path"],
            )
            for b in made
        ]
        log.info("uploading %d audio bunch(es) packed from %d shard(s)…", len(made), len(files))
        _commit_with_backoff(
            lambda: api.create_commit(
                repo_id=cfg.repos.data, repo_type="dataset", operations=ops,
                commit_message=f"audio: {len(made)} bunches from {len(files)} shards",
            ),
            "bunch upload",
        )
        for f in files:
            try:
                idx = int(f[len("shard_"):-len(".parquet")])
                upsert_shard(state, idx, rows=0, uploaded=True)
            except ValueError:
                pass
            try:
                os.remove(os.path.join(pdir, f))
            except OSError:
                pass
        state["bunched"] = True
        state["next_bunch_index"] = start + len(made)
        state.setdefault("bunches", [])
        for b in made:
            state["bunches"].append({
                "index": b["index"], "rows": b["rows"], "uploaded": True,
                "hub_path": f"{HUB_SHARD_DIR}/bunch_{b['index']:05d}.parquet",
            })
            try:
                os.remove(b["path"])
            except OSError:
                pass
        n_sent = len(made)
    else:
        n_sent = 0

    persist_state(cfg, state, seen)
    ls, lf = local_paths(cfg.paths.audio_dir)
    _commit_with_backoff(
        lambda: api.create_commit(
            repo_id=cfg.repos.data, repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=HUB_STATE, path_or_fileobj=ls),
                        CommitOperationAdd(path_in_repo=HUB_FPS, path_or_fileobj=lf)],
            commit_message="update resume state after bunch upload",
        ),
        "state commit",
    )
    log.info("bunch upload complete: %d file(s) -> %s", n_sent, cfg.repos.data)
    return n_sent


def stage_shard(cfg, arrow_dir: str, idx: int, *, drop_local: bool) -> tuple[str, int]:
    """Convert one arrow shard to parquet on local disk (no Hub commit yet)."""
    parquet_path = os.path.join(cfg.paths.audio_dir, "parquet_tmp", f"shard_{idx:05d}.parquet")
    rows = arrow_to_parquet(arrow_dir, parquet_path)
    if drop_local:
        shutil.rmtree(arrow_dir, ignore_errors=True)
    return parquet_path, rows


def commit_batch(cfg, state: dict, seen: set[str], staged: list[tuple[int, str, int]]) -> None:
    """Upload MANY staged parquet shards + resume state in ONE Hub commit.

    Batching keeps us far under the Hub's 128-commits/hour limit. On 429 we back
    off with increasing sleeps; a final failure raises so resume retries later.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    if not staged:
        return
    persist_state(cfg, state, seen)
    local_state_path, local_fps_path = local_paths(cfg.paths.audio_dir)
    ops = [CommitOperationAdd(path_in_repo=f"{HUB_SHARD_DIR}/shard_{idx:05d}.parquet",
                              path_or_fileobj=p) for idx, p, _ in staged]
    ops.append(CommitOperationAdd(path_in_repo=HUB_STATE, path_or_fileobj=local_state_path))
    ops.append(CommitOperationAdd(path_in_repo=HUB_FPS, path_or_fileobj=local_fps_path))

    api = HfApi(token=require_token())
    lo, hi = staged[0][0], staged[-1][0]
    msg = f"audio shards {lo:05d}..{hi:05d} ({len(staged)} files)"
    delays = [0, 30, 120, 300, 900, 1800]
    for attempt, delay in enumerate(delays):
        if delay:
            log.warning("commit backoff %ds (attempt %d/%d)…", delay, attempt + 1, len(delays))
            time.sleep(delay)
        try:
            api.create_commit(repo_id=cfg.repos.data, repo_type="dataset",
                              operations=ops, commit_message=msg)
            break
        except Exception as e:
            rate = "429" in str(e) or "Too Many Requests" in str(e)
            log.warning("batch commit failed%s: %s", " (rate-limited)" if rate else "", e)
            if attempt == len(delays) - 1:
                raise RuntimeError(f"commit_batch {msg} failed after retries") from e

    for idx, p, rows in staged:
        upsert_shard(state, idx, rows, uploaded=True)
        try:
            os.remove(p)
        except OSError:
            pass
    persist_state(cfg, state, seen)
    log.info("HUB COMMIT ok: %d shards (%05d..%05d) -> %s", len(staged), lo, hi, cfg.repos.data)
    log.info("resume  %s", format_progress(state, cfg.data.max_hours_per_lang))


def upload_arrow_shard(cfg, state: dict, seen: set[str], idx: int, arrow_dir: str,
                       *, drop_local: bool) -> None:
    import time
    import shutil

    parquet_path = os.path.join(cfg.paths.audio_dir, "parquet_tmp", f"shard_{idx:05d}.parquet")
    rows = arrow_to_parquet(arrow_dir, parquet_path)
    upsert_shard(state, idx, rows, uploaded=False)
    last_err = None
    for attempt in range(1, 4):
        try:
            commit_shard(cfg, state, seen, parquet_path, idx, rows)
            last_err = None
            break
        except Exception as e:
            last_err = e
            log.warning("upload shard_%05d attempt %d/3 failed: %s", idx, attempt, e)
            time.sleep(2 ** attempt)
    if last_err is not None:
        raise RuntimeError(f"failed to upload shard_{idx:05d} after 3 tries") from last_err
    try:
        os.remove(parquet_path)
    except OSError:
        pass
    if drop_local:
        shutil.rmtree(arrow_dir, ignore_errors=True)
        log.info("removed local %s (on Hub now)", arrow_dir)
