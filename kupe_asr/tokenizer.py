"""Extend the Gemma-3 tokenizer with audio + control tokens, and expose the id map.

The audio tokens are added as ONE contiguous block, so a Mimi code `c` maps to a
single id `mimi_base + c`. We assert contiguity so a silent tokenizer change can
never corrupt the audio embedding lookup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import constants as C
from .hf_utils import hf_token, log


@dataclass
class TokenMap:
    """Everything the collator / streamer needs to build sequences, computed once."""
    bos: int
    eos: int
    pad: int
    audio_start: int
    audio_end: int
    lang_auto: int
    transcribe: int
    lang: dict[str, int]      # code -> token id, e.g. {"hi": 2620...}
    lang_id_to_code: dict[int, str]
    mimi_base: int            # id of <|mimi_0|>
    vocab_size: int


def build_tokenizer(base_id: str, save_dir: str):
    """Load Gemma tokenizer, inject our tokens, save to `save_dir`. Returns tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_id, token=hf_token())

    # A pad token is required for right-padding batches.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Control tokens are "special" (never split); audio tokens are plain added
    # tokens (also atomic, but not flagged special so they don't clutter decode).
    n_added = tok.add_special_tokens({"additional_special_tokens": C.CONTROL_TOKENS})
    n_added += tok.add_tokens(C.AUDIO_TOKENS)
    log.info("added %d tokens (%d control + %d audio)", n_added,
             len(C.CONTROL_TOKENS), len(C.AUDIO_TOKENS))

    os.makedirs(save_dir, exist_ok=True)
    tok.save_pretrained(save_dir)
    log.info("tokenizer saved -> %s (vocab=%d)", save_dir, len(tok))
    return tok


def load_tokenizer(save_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(save_dir)


def token_map(tok) -> TokenMap:
    """Derive the id map from an (already extended) tokenizer, with sanity checks."""
    mimi_base = tok.convert_tokens_to_ids(C.mimi_token(0))
    mimi_last = tok.convert_tokens_to_ids(C.mimi_token(C.MIMI_CODEBOOK_SIZE - 1))
    assert mimi_base != tok.unk_token_id, "audio tokens missing — rebuild tokenizer"
    assert mimi_last == mimi_base + C.MIMI_CODEBOOK_SIZE - 1, (
        "audio token ids are not contiguous — rebuild tokenizer"
    )

    lang = {c: tok.convert_tokens_to_ids(C.lang_token(c)) for c in C.LANG_CODES}
    return TokenMap(
        bos=tok.bos_token_id,
        eos=tok.eos_token_id,
        pad=tok.pad_token_id,
        audio_start=tok.convert_tokens_to_ids(C.AUDIO_START),
        audio_end=tok.convert_tokens_to_ids(C.AUDIO_END),
        lang_auto=tok.convert_tokens_to_ids(C.LANG_AUTO),
        transcribe=tok.convert_tokens_to_ids(C.TRANSCRIBE),
        lang=lang,
        lang_id_to_code={v: k for k, v in lang.items()},
        mimi_base=mimi_base,
        vocab_size=len(tok),
    )
