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
        name="commonvoice",
        hf_id="fsicoli/common_voice_17_0",     # parquet mirror; mozilla's script is gone
        lang_configs=_ISO_CFG,                 # en, hi, gu, bn, mr, ur
        splits=["train", "validation"],
        trust_remote_code=True,
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


def _open_stream(source: Source, cfg: str, token: str):
    """Try each split for one config; return (IterableDataset, split) or raise."""
    from datasets import load_dataset

    last = None
    for split in source.splits:
        try:
            ds = load_dataset(
                source.hf_id, cfg, split=split, streaming=True,
                token=token, trust_remote_code=source.trust_remote_code,
            )
            return ds, split
        except Exception as e:  # split may not exist for this config
            last = e
            continue
    raise SkipSource(f"no readable split for {source.hf_id}:{cfg} ({last})")


def iter_examples(source: Source, lang: str, token: str) -> Iterator[tuple]:
    """Yield (audio_array, sampling_rate, text) for one (source, lang).

    Raises SkipSource before yielding anything if the source can't be opened.
    Per-row decode errors are swallowed so one bad clip never stops the stream.
    """
    from datasets import Audio

    cfgs = source.lang_configs[lang]
    ds = split = None
    for cfg in cfgs:
        try:
            ds, split = _open_stream(source, cfg, token)
            log.info("open %s [%s] split=%s (lang=%s)", source.name, cfg, split, lang)
            break
        except SkipSource as e:
            log.debug("cfg %s failed: %s", cfg, e)
    if ds is None:
        raise SkipSource(f"{source.name}: no working config for '{lang}' ({cfgs})")

    if not ds.features:
        raise SkipSource(f"{source.name}:{lang} exposes no feature schema")
    audio_col = _find_audio_col(ds.features)
    text_col = _find_text_col(ds.features)
    # ensure decode is on (some streams ship raw bytes)
    ds = ds.cast_column(audio_col, Audio())

    it = iter(ds)
    while True:
        try:
            row = next(it)
        except StopIteration:
            return
        except Exception as e:            # stream died -> surface it, stop this source
            log.warning("stream %s/%s ended early: %s", source.name, lang, e)
            return
        try:
            a = row[audio_col]
            text = row[text_col]
            if a is None or not text or not str(text).strip():
                continue
            yield a["array"], a["sampling_rate"], str(text).strip()
        except Exception as e:            # corrupt clip -> skip, keep going
            log.debug("row skipped in %s/%s: %s", source.name, lang, e)
            continue
