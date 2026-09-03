"""Stage 1 — stream verified sources, resample to 24 kHz mono, shard to parquet.

Streaming + per-language hour caps means we only download what we keep. Sources
are interleaved round-robin per language for a balanced accent/domain mix.
Output: local `audio` dataset (save_to_disk) + pushed `audio` config on the Hub.
"""
from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from ..constants import LANGUAGES, MIMI_SAMPLE_RATE
from ..hf_utils import log, require_token
from ..text import is_probably_valid, normalize
from .shards import ShardWriter, load_all, wav_bytes
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
    if array.ndim > 1:                      # stereo -> mono (channels = smallest axis)
        ch_axis = int(np.argmin(array.shape))
        array = np.asarray(array.mean(axis=ch_axis), dtype=np.float32).reshape(-1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(array, dtype=np.float32)


def fetch(cfg) -> str:
    import datasets

    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bars()          # kill "Resolving data files" spam

    token = require_token()
    target_sr = int(cfg.data.target_sr) if hasattr(cfg.data, "target_sr") else MIMI_SAMPLE_RATE
    shards_dir = os.path.join(cfg.paths.audio_dir, "shards")
    writer = ShardWriter(shards_dir, _features(target_sr), int(cfg.data.shard_size))

    kept = {lang: 0.0 for lang in cfg.languages}
    caps = cfg.data.max_hours_per_lang

    for lang in cfg.languages:
        cap_s = float(getattr(caps, lang, 0)) * 3600.0
        if cap_s <= 0:
            continue
        val_cap_s = float(cfg.data.val_minutes_per_lang) * 60.0
        got_s = 0.0
        val_s = 0.0
        n = 0

        iters = []
        by_src: dict[str, int] = {}
        for src in sources_for(lang):
            try:
                iters.append([src.name, iter_examples(src, lang, token)])
                by_src[src.name] = 0
            except SkipSource as e:
                log.warning("skip %s/%s: %s", src.name, lang, e)

        if not iters:
            log.warning("no sources available for %s — skipping", lang)
            continue

        active = iters[:]
        pbar = tqdm(total=cap_s, unit="s", unit_scale=True,
                    desc=f"fetch {lang} ({LANGUAGES[lang]})")
        while active and got_s < cap_s:
            for entry in list(active):
                if got_s >= cap_s:
                    break
                name, it = entry
                try:
                    array, sr, text = next(it)
                except StopIteration:
                    active.remove(entry)
                    continue
                except Exception as e:               # keep the stream alive
                    log.debug("next() error %s/%s: %s", name, lang, e)
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

                split = "val" if val_s < val_cap_s else "train"
                writer.add({
                    "id": f"{lang}-{name}-{n:08d}",
                    "language": lang,
                    "source": name,
                    "text": normalize(text),
                    "duration": float(dur),
                    "split": split,
                    "audio": {"bytes": wav_bytes(wav, target_sr), "path": None},
                })
                n += 1
                by_src[name] = by_src.get(name, 0) + 1
                got_s += dur
                if split == "val":
                    val_s += dur
                pbar.update(dur)
        pbar.close()
        kept[lang] = got_s
        log.info("lang %s: kept %.1f h (%d clips, %.1f min val) | by source: %s",
                 lang, got_s / 3600, n, val_s / 60, by_src)
        if n == 0:
            log.warning("lang %s produced 0 clips — check gate/access for its sources", lang)

    shard_paths = writer.close()
    ds = load_all(shard_paths)

    out_dir = os.path.join(cfg.paths.audio_dir, "dataset")
    ds.save_to_disk(out_dir)
    log.info("audio dataset saved -> %s (%d rows)", out_dir, ds.num_rows)
    log.info("hours per language: %s", {k: round(v / 3600, 1) for k, v in kept.items()})

    if cfg.data.push:
        ds.push_to_hub(cfg.repos.data, config_name="audio", token=token,
                       commit_message="add audio config (raw 24kHz)")
        log.info("pushed `audio` config -> %s", cfg.repos.data)

    return out_dir
