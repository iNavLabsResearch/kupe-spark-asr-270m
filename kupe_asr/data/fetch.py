"""Stage 1 — stream sources, shard, upload each shard to Hub, resume later.

Each flushed shard is converted to parquet and committed immediately with
`audio/fetch_state.json` so a crash does not lose Hub progress.

On restart:
  1. index leftover local shards
  2. upload any shard not yet on Hub
  3. continue fetching from the hour caps in fetch_state.json
"""
from __future__ import annotations

import gc
import os
import shutil

import numpy as np
from tqdm import tqdm


def _free_memory() -> None:
    """Return freed heap to the OS. Audio+Arrow loops fragment glibc's allocator,
    so RSS creeps up and OOM-kills on small boxes without this."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

from ..constants import LANGUAGES, MIMI_SAMPLE_RATE
from ..hf_utils import ensure_repo, log, require_token
from ..text import is_probably_valid, normalize
from .fetch_state import (
    clip_fp,
    commit_batch,
    empty_lang,
    ensure_data_card,
    ingest_local_shard,
    list_local_shards,
    load_merged_state,
    persist_state,
    print_status,
    stage_shard,
    upload_arrow_shard,
    upsert_shard,
)
from .shards import ShardWriter, wav_bytes
from .sources import SkipSource, iter_examples, sources_for


def _features(target_sr: int):
    from datasets import Audio, Features, Value

    return Features({
        "id": Value("string"),
        "language": Value("string"),
        "source": Value("string"),
        "text": Value("string"),
        "duration": Value("float32"),
        "split": Value("string"),
        "audio": Audio(sampling_rate=target_sr),
    })


def _to_mono_24k(array, sr: int, target_sr: int) -> np.ndarray:
    import librosa

    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        ch_axis = int(np.argmin(array.shape))
        array = np.asarray(array.mean(axis=ch_axis), dtype=np.float32).reshape(-1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(array, dtype=np.float32)


def _counted_shard_indices(state: dict, hub_indices: set[int]) -> set[int]:
    counted = set(hub_indices)
    has_counts = any(
        int((st or {}).get("n_clips", 0)) > 0
        for st in (state.get("languages") or {}).values()
    )
    for s in state.get("shards") or []:
        idx = int(s["index"])
        if int(s.get("rows") or 0) > 0 or has_counts:
            counted.add(idx)
    return counted


def _index_and_upload_local(cfg, state: dict, seen: set[str], hub_indices: set[int],
                            shards_dir: str, *, push: bool, hub_only: bool) -> None:
    local = list_local_shards(shards_dir)
    if local:
        log.info("found %d local shard(s) under %s", len(local), shards_dir)
    counted = _counted_shard_indices(state, hub_indices)

    for idx, path in local:
        if idx not in counted:
            ingest_local_shard(state, list(cfg.languages), idx, path, seen)
            counted.add(idx)
        else:
            upsert_shard(state, idx, rows=int(next(
                (s.get("rows") or 0 for s in state["shards"] if int(s["index"]) == idx), 0
            )), uploaded=idx in hub_indices)

        already = idx in hub_indices or any(
            int(s["index"]) == idx and s.get("uploaded") for s in state["shards"]
        )
        if already:
            log.info("shard_%05d already on Hub — skip upload", idx)
            upsert_shard(state, idx, rows=0, uploaded=True)
            if hub_only:
                shutil.rmtree(path, ignore_errors=True)
                log.info("removed local shard_%05d (already on Hub)", idx)
            continue
        if not push:
            log.info("shard_%05d local only (--no-push)", idx)
            continue
        # stage only; _recover_parquet_tmp commits leftovers in ONE batched commit
        log.info("staging leftover local shard_%05d for batched upload…", idx)
        stage_shard(cfg, path, idx, drop_local=hub_only)

    persist_state(cfg, state, seen)


def _recover_parquet_tmp(cfg, state, seen, hub_indices, batch_n) -> None:
    """Commit parquet shards staged locally but not committed before a crash."""
    import pyarrow.parquet as pq

    tmp_dir = os.path.join(cfg.paths.audio_dir, "parquet_tmp")
    if not os.path.isdir(tmp_dir):
        return
    staged: list[tuple[int, str, int]] = []
    for name in sorted(os.listdir(tmp_dir)):
        if not (name.startswith("shard_") and name.endswith(".parquet")):
            continue
        try:
            idx = int(name[len("shard_"):-len(".parquet")])
        except ValueError:
            continue
        path = os.path.join(tmp_dir, name)
        if idx in hub_indices:
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        try:
            rows = pq.ParquetFile(path).metadata.num_rows
        except Exception:
            continue
        staged.append((idx, path, rows))
        if len(staged) >= batch_n:
            commit_batch(cfg, state, seen, staged)
            for i, _, _ in staged:
                hub_indices.add(i)
            staged.clear()
    if staged:
        log.info("recovering %d staged shard(s) from a previous run", len(staged))
        commit_batch(cfg, state, seen, staged)
        for i, _, _ in staged:
            hub_indices.add(i)


def fetch_status(cfg, *, reset: bool = False) -> dict:
    token = require_token()
    ensure_repo(cfg.repos.data, "dataset")
    state, seen, hub_indices = load_merged_state(cfg, reset=reset)
    shards_dir = os.path.join(cfg.paths.audio_dir, "shards")
    counted = _counted_shard_indices(state, hub_indices)
    for idx, path in list_local_shards(shards_dir):
        if idx not in counted:
            ingest_local_shard(state, list(cfg.languages), idx, path, seen)
    print_status(state, cfg.data.max_hours_per_lang, hub_indices)
    persist_state(cfg, state, seen)
    return state


def fetch(cfg, *, hub_only: bool = False, reset: bool = False) -> str:
    import datasets

    datasets.utils.logging.set_verbosity_error()
    # keep tqdm bars ON (download/resolve/upload MB bars); only httpx logs are silenced.

    token = require_token()
    if hub_only and not cfg.data.push:
        raise RuntimeError("--hub-only needs data.push=true; do not pass --no-push")

    ensure_repo(cfg.repos.data, "dataset")
    if cfg.data.push:
        ensure_data_card(cfg)

    target_sr = int(cfg.data.target_sr) if hasattr(cfg.data, "target_sr") else MIMI_SAMPLE_RATE
    shards_dir = os.path.join(cfg.paths.audio_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    state, seen, hub_indices = load_merged_state(cfg, reset=reset)
    _index_and_upload_local(
        cfg, state, seen, hub_indices, shards_dir,
        push=bool(cfg.data.push), hub_only=hub_only,
    )
    if cfg.data.push:
        _recover_parquet_tmp(cfg, state, seen, hub_indices,
                             int(getattr(cfg.data, "commit_batch", 25)))
    print_status(state, cfg.data.max_hours_per_lang, hub_indices)

    start_index = int(state.get("next_shard_index") or 0)
    for s in state.get("shards") or []:
        start_index = max(start_index, int(s["index"]) + 1)
    if hub_indices:
        start_index = max(start_index, max(hub_indices) + 1)

    # Collect several shards locally, then upload them in ONE Hub commit.
    # HF caps commits at 128/hour; one-commit-per-shard hits 429 fast.
    staged: list[tuple[int, str, int]] = []
    batch_n = int(getattr(cfg.data, "commit_batch", 25))

    def flush_staged(force: bool = False) -> None:
        if staged and (force or len(staged) >= batch_n):
            commit_batch(cfg, state, seen, staged)
            for idx, _, _ in staged:
                hub_indices.add(idx)
            staged.clear()

    def on_flush(arrow_dir: str, idx: int, nrows: int) -> None:
        upsert_shard(state, idx, nrows, uploaded=False)
        persist_state(cfg, state, seen)
        if not cfg.data.push:
            log.info("shard_%05d saved locally (%d rows) — not uploaded (--no-push)", idx, nrows)
            return
        pq_path, rows = stage_shard(cfg, arrow_dir, idx, drop_local=hub_only)
        staged.append((idx, pq_path, rows))
        log.info("staged shard_%05d (%d rows) — %d/%d before next commit", idx, rows, len(staged), batch_n)
        flush_staged()
        _free_memory()          # keep RSS flat across many shards

    writer = ShardWriter(
        shards_dir, _features(target_sr), int(cfg.data.shard_size),
        start_index=start_index, on_flush=on_flush,
    )

    caps = cfg.data.max_hours_per_lang

    for lang in cfg.languages:
        cap_s = float(getattr(caps, lang, 0)) * 3600.0
        if cap_s <= 0:
            continue
        st = state["languages"].setdefault(lang, empty_lang())
        got_s = float(st.get("kept_seconds") or 0)
        val_s = float(st.get("val_seconds") or 0)
        n = int(st.get("next_clip_index") or st.get("n_clips") or 0)
        val_cap_s = float(cfg.data.val_minutes_per_lang) * 60.0
        by_src = dict(st.get("by_source") or {})

        if got_s >= cap_s:
            st["status"] = "done"
            log.info("skip %s — already have %.1f h (cap %.1f h)", lang, got_s / 3600, cap_s / 3600)
            continue

        st["status"] = "in_progress"
        persist_state(cfg, state, seen)

        files_done = dict(st.get("files_done") or {})
        iters = []
        for src in sources_for(lang):
            try:
                sf0 = int(files_done.get(src.name, 0))
                iters.append([src.name, iter_examples(src, lang, token, start_file=sf0)])
                by_src.setdefault(src.name, 0)
            except SkipSource as e:
                log.warning("skip %s/%s: %s", src.name, lang, e)

        if not iters:
            log.warning("no sources available for %s — skipping", lang)
            st["status"] = "done"
            continue

        dups = 0
        pbar = tqdm(total=cap_s / 3600.0, initial=got_s / 3600.0, unit="h",
                    desc=f"fetch {lang} ({LANGUAGES[lang]})",
                    bar_format="{l_bar}{bar}| {n:.2f}/{total:.1f}h "
                               "[{elapsed}<{remaining}] {postfix}")
        for name, it in iters:                # drain one source fully before the next
            if got_s >= cap_s:
                break
            while got_s < cap_s:
                try:
                    array, sr, text, file_idx = next(it)
                except StopIteration:
                    break
                except Exception as e:
                    log.debug("next() error %s/%s: %s", name, lang, e)
                    continue

                if array is None:            # a parquet file finished -> save resume point
                    files_done[name] = int(file_idx)
                    st["files_done"] = files_done
                    persist_state(cfg, state, seen)
                    _free_memory()
                    continue

                if not is_probably_valid(text):
                    continue
                try:
                    wav = _to_mono_24k(array, sr, target_sr)
                except Exception as e:
                    log.debug("resample fail: %s", e)
                    continue
                dur = len(wav) / target_sr
                if dur < cfg.data.min_dur or dur > cfg.data.max_dur:
                    continue

                text_n = normalize(text)
                fp = clip_fp(lang, name, text_n, dur)
                if fp in seen:
                    dups += 1
                    if dups % 500 == 0:
                        log.info("resume %s: skipped %d clips already in shards/Hub", lang, dups)
                    continue

                split = "val" if val_s < val_cap_s else "train"
                writer.add({
                    "id": f"{lang}-{name}-{n:08d}",
                    "language": lang,
                    "source": name,
                    "text": text_n,
                    "duration": float(dur),
                    "split": split,
                    "audio": {"bytes": wav_bytes(wav, target_sr), "path": None},
                })
                seen.add(fp)
                n += 1
                by_src[name] = by_src.get(name, 0) + 1
                got_s += dur
                if split == "val":
                    val_s += dur
                st["kept_seconds"] = got_s
                st["val_seconds"] = val_s
                st["n_clips"] = int(st.get("n_clips") or 0) + 1
                st["next_clip_index"] = n
                st["by_source"] = by_src
                pbar.update(dur / 3600.0)
                if n % 200 == 0:
                    pbar.set_postfix_str(f"{n} clips · {by_src}")
        pbar.close()
        st["kept_seconds"] = got_s
        st["val_seconds"] = val_s
        st["n_clips"] = int(st.get("n_clips") or 0)
        st["next_clip_index"] = n
        st["by_source"] = by_src
        st["files_done"] = files_done
        # reached here = cap hit or all sources exhausted -> nothing more to fetch
        st["status"] = "done"
        persist_state(cfg, state, seen)
        log.info("lang %s: kept %.1f h (%d clips, %.1f min val) | by source: %s | %s",
                 lang, got_s / 3600, st["n_clips"], val_s / 60, by_src, st["status"])
        if st["n_clips"] == 0:
            log.warning("lang %s produced 0 clips — check gate/access for its sources", lang)

    writer.close()
    flush_staged(force=True)          # upload any remaining staged shards in one commit
    local_state, local_fps = persist_state(cfg, state, seen)
    if cfg.data.push:
        from huggingface_hub import CommitOperationAdd, HfApi
        try:
            HfApi(token=token).create_commit(
                repo_id=cfg.repos.data,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo="audio/fetch_state.json", path_or_fileobj=local_state),
                    CommitOperationAdd(path_in_repo="audio/seen_fps.txt", path_or_fileobj=local_fps),
                ],
                commit_message="fetch complete — update resume state",
            )
        except Exception as e:
            log.warning("final state commit skipped: %s", e)

    hours = {k: round(float((state["languages"].get(k) or {}).get("kept_seconds") or 0) / 3600, 1)
             for k in cfg.languages}
    n_up = sum(1 for s in state.get("shards") or [] if s.get("uploaded"))
    log.info("fetch done | hours=%s | shards uploaded=%d | repo=%s [audio]",
             hours, n_up, cfg.repos.data)
    return cfg.repos.data if hub_only or cfg.data.push else os.path.join(cfg.paths.audio_dir, "shards")
