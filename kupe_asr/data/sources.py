"""Verified HF ASR sources for our 6 languages, plus a robust streaming reader.

Design choices that keep a fresh GPU box from crashing:
  * every (source, language) is tried independently; a failure is logged & skipped,
    never fatal (gated-not-accepted, missing config, dead split, etc.);
  * config names are tried from a candidate list (datasets disagree on casing);
  * the audio and text columns are auto-detected from the dataset features, because
    IndicVoices / Vaani do not document their exact column names.

Sources (all fact-checked on the Hub, 2026-09):
  ai4bharat/IndicVoices               gated, parquet, configs = language names
  ARTPARK-IISc/Vaani-transcription-part  gated, parquet, configs = language names
  mozilla-foundation/common_voice_17_0   gated, script, configs = ISO codes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ..hf_utils import log


class SkipSource(Exception):
    """Raised internally when a (source, lang) cannot be read; caller skips it."""


# candidate config spellings per language for the "language-name" datasets
# (IndicVoices/Vaani use lowercase language names -> try lowercase first)
_NAME_CFG = {
    "hi": ["hindi", "Hindi"],
    "gu": ["gujarati", "Gujarati"],
    "bn": ["bengali", "Bengali"],
    "mr": ["marathi", "Marathi"],
    "ur": ["urdu", "Urdu"],
    "en": ["english", "English"],
}
_ISO_CFG = {c: [c] for c in ["en", "hi", "gu", "bn", "mr", "ur"]}

_TEXT_COL_CANDIDATES = [
    "text", "sentence", "transcript", "transcription", "normalized_text",
    "normalized", "verbatim", "transcript_normalized", "raw_text", "target",
]


@dataclass
class Source:
    name: str
    hf_id: str
    lang_configs: dict[str, list[str]]   # our code -> candidate config names
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "valid"])
    trust_remote_code: bool = False


# Indic core: AI4Bharat + ARTPARK-IISc (parquet, no script).
# English: fsicoli CV mirror (script) + People's Speech (script-free backstop).
SOURCES: list[Source] = [
    Source(
        name="indicvoices",
        hf_id="ai4bharat/IndicVoices",
        lang_configs={k: _NAME_CFG[k] for k in ["hi", "gu", "bn", "mr", "ur"]},
        splits=["train", "valid", "validation"],
    ),
    Source(
        name="vaani",
        hf_id="ARTPARK-IISc/Vaani-transcription-part",
        lang_configs={k: _NAME_CFG[k] for k in ["hi", "gu", "bn", "mr", "ur"]},
        splits=["train"],
    ),
    Source(
        name="peoples_speech",
        hf_id="MLCommons/peoples_speech",       # 30k h English, CC-BY, no gate
        lang_configs={"en": ["clean", "dirty"]},
        splits=["train"],
        trust_remote_code=True,
    ),
]


def sources_for(lang: str) -> list[Source]:
    return [s for s in SOURCES if lang in s.lang_configs]


def _find_audio_col(features) -> str:
    from datasets import Audio

    for name, feat in features.items():
        if isinstance(feat, Audio):
            return name
    # fall back to a column literally called "audio"
    if "audio" in features:
        return "audio"
    raise SkipSource("no Audio column found")


def _find_text_col(features) -> str:
    lower = {k.lower(): k for k in features}
    for cand in _TEXT_COL_CANDIDATES:
        if cand in lower:
            return lower[cand]
    raise SkipSource(f"no text column among {list(features)}")


def _decode_audio(a):
    """Return (float32 array, sampling_rate) from any HF audio cell shape.

    Handles decoded Audio ({'array','sampling_rate'}) and raw structs
    ({'bytes': ...} / {'path': ...}) so we never rely on a lazy cast_column.
    """
    import io

    import numpy as np

    if a is None:
        return None, None
    if isinstance(a, dict):
        if a.get("array") is not None and a.get("sampling_rate"):
            return np.asarray(a["array"], dtype=np.float32), int(a["sampling_rate"])
        import soundfile as sf

        if a.get("bytes"):
            arr, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
            return arr, sr
        if a.get("path"):
            arr, sr = sf.read(a["path"], dtype="float32", always_2d=False)
            return arr, sr
    return None, None


def _list_parquet_files(hf_id: str, cfg: str, splits: list[str], token: str) -> list[str]:
    """Sorted parquet file paths for one config, split-preferred if possible."""
    from huggingface_hub import HfApi

    files = HfApi(token=token).list_repo_files(hf_id, repo_type="dataset")
    pref = cfg.rstrip("/") + "/"
    cand = [f for f in files if f.endswith(".parquet") and
            (f.startswith(pref) or ("/" + pref) in ("/" + f))]
    for sp in splits:                       # prefer files whose name carries the split
        sub = [f for f in cand if sp in f.rsplit("/", 1)[-1]]
        if sub:
            return sorted(sub)
    return sorted(cand)


def _resolve_files(source: Source, lang: str, token: str) -> tuple[str, list[str]]:
    for cfg in source.lang_configs[lang]:
        try:
            files = _list_parquet_files(source.hf_id, cfg, source.splits, token)
        except Exception as e:
            log.debug("list %s:%s failed: %s", source.hf_id, cfg, e)
            files = []
        if files:
            return cfg, files
    raise SkipSource(f"{source.name}: no parquet files for '{lang}' ({source.lang_configs[lang]})")


def _cols_from_schema(schema) -> tuple[str | None, str | None]:
    import pyarrow as pa

    lower = {n.lower(): n for n in schema.names}
    text = next((lower[c] for c in _TEXT_COL_CANDIDATES if c in lower), None)
    audio = None
    for n in schema.names:
        t = schema.field(n).type
        if n == "audio" or (pa.types.is_struct(t) and
                            any(t.field(i).name == "bytes" for i in range(t.num_fields))):
            audio = n
            break
    return audio, text


def iter_examples(source: Source, lang: str, token: str, start_file: int = 0) -> Iterator[tuple]:
    """Yield (array, sampling_rate, text, file_idx) per clip, and a
    (None, None, None, file_idx+1) marker after each file is fully consumed.

    Progress is tracked per whole parquet FILE, so resume starts at `start_file`
    and never re-reads finished files (no O(n) row skip). One file is downloaded
    at a time to a temp dir and deleted after use -> bounded RAM + disk.
    """
    import os
    import tempfile

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    cfg, files = _resolve_files(source, lang, token)
    log.info("open %s [%s]: %d parquet files, start at %d (lang=%s)",
             source.name, cfg, len(files), start_file, lang)

    for i in range(start_file, len(files)):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                local = hf_hub_download(source.hf_id, files[i], repo_type="dataset",
                                        token=token, local_dir=tmp)
            except Exception as e:
                log.warning("download %s failed: %s", files[i], e)
                yield None, None, None, i + 1
                continue
            try:
                pf = pq.ParquetFile(local)
                audio_col, text_col = _cols_from_schema(pf.schema_arrow)
                if not audio_col or not text_col:
                    log.warning("cols not found in %s (audio=%s text=%s)", files[i], audio_col, text_col)
                else:
                    for batch in pf.iter_batches(batch_size=64):
                        for rec in batch.to_pylist():
                            try:
                                text = rec.get(text_col)
                                if not text or not str(text).strip():
                                    continue
                                arr, sr = _decode_audio(rec.get(audio_col))
                                if arr is None:
                                    continue
                                yield arr, sr, str(text).strip(), i
                            except Exception as e:
                                log.debug("row skip %s: %s", files[i], e)
                                continue
            except Exception as e:
                log.warning("read %s failed: %s", files[i], e)
        yield None, None, None, i + 1        # file i done -> caller sets files_done=i+1
