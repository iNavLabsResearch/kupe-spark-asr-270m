"""Text normalisation shared by data prep and evaluation.

Deliberately light and script-agnostic (NFC + punctuation strip + whitespace
collapse). WER/CER are computed on this normalised form so scoring is consistent
across sources. It is NOT a linguistic normaliser — good enough and reproducible.
"""
from __future__ import annotations

import re
import unicodedata

# punctuation across Latin + Indic + Arabic scripts
_PUNCT = re.compile(
    r"[.,!?;:\"'`~^*_=+\\/<>@#$%&(){}\[\]|।॥،؛؟…“”‘’—–\-]"
)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()                 # no-op for Indic scripts, helps Latin
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def is_probably_valid(text: str, min_chars: int = 1) -> bool:
    t = normalize(text)
    return len(t) >= min_chars and any(ch.isalnum() for ch in t)
