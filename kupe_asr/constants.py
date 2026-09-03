"""Static contract shared by every stage: languages, special tokens, codec geometry.

Nothing here reads config or env. Import this anywhere without side effects.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Languages we actually teach the model. Order is stable and used everywhere.
# code -> human name. Keep this list closed; unsupported scripts are ignored.
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "ur": "Urdu",
    "mr": "Marathi",
}
LANG_CODES: list[str] = list(LANGUAGES.keys())

# ---------------------------------------------------------------------------
# Mimi codec geometry (kyutai/mimi). Verified from the model config.
# ---------------------------------------------------------------------------
MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_RATE = 12.5
MIMI_SAMPLES_PER_FRAME = int(MIMI_SAMPLE_RATE / MIMI_FRAME_RATE)  # 1920
MIMI_CODEBOOK_SIZE = 2048  # codes per codebook; we use codebook 0 (semantic)

# ---------------------------------------------------------------------------
# Special tokens added on top of the Gemma tokenizer.
# ---------------------------------------------------------------------------
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
LANG_AUTO = "<|lang_auto|>"          # "detect the language yourself"
TRANSCRIBE = "<|transcribe|>"        # reserved task tag (future multi-task)


def lang_token(code: str) -> str:
    """Control token that forces / marks a language, e.g. 'hi' -> '<|lang_hi|>'."""
    return f"<|lang_{code}|>"


def mimi_token(code: int) -> str:
    """Audio token string for a Mimi codebook-0 code id."""
    return f"<|mimi_{code}|>"


# Full, ordered list of the special (non-audio) tokens.
CONTROL_TOKENS: list[str] = (
    [AUDIO_START, AUDIO_END, LANG_AUTO, TRANSCRIBE]
    + [lang_token(c) for c in LANG_CODES]
)

# Audio tokens are added as ONE contiguous block so their ids stay contiguous.
AUDIO_TOKENS: list[str] = [mimi_token(i) for i in range(MIMI_CODEBOOK_SIZE)]

# All tokens we inject, in the exact order they must be added.
ADDED_TOKENS: list[str] = CONTROL_TOKENS + AUDIO_TOKENS
