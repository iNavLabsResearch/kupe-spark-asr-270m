"""Sequence builder + dynamic-padding collator.

Sequence layout (causal LM):

  forced mode  : <bos><audio_start> a0..aT <audio_end> <lang_XX>  text.. <eos>
  auto mode    : <bos><audio_start> a0..aT <audio_end> <lang_auto> <lang_XX> text.. <eos>
                 |------------------ prefix (label=-100) ---------|-- supervised --|

In auto mode the model must first emit the detected <lang_XX> token, then the
text — that single token IS the language-id prediction at inference.
"""
from __future__ import annotations

import random

import torch

from ..tokenizer import TokenMap


class SequenceBuilder:
    def __init__(self, tokenizer, tmap: TokenMap, max_seq_len: int,
                 max_audio_frames: int, max_text_tokens: int):
        self.tok = tokenizer
        self.m = tmap
        self.max_seq_len = max_seq_len
        self.max_audio_frames = max_audio_frames
        self.max_text_tokens = max_text_tokens

    def build(self, mimi_cb0, text: str, lang: str, mode: str):
        m = self.m
        audio = [m.mimi_base + int(c) for c in mimi_cb0[: self.max_audio_frames]]
        text_ids = self.tok(text, add_special_tokens=False).input_ids[: self.max_text_tokens]
        lang_id = m.lang[lang]

        if mode == "forced":
            control = [m.lang[lang]]
            target = text_ids + [m.eos]
        else:  # auto
            control = [m.lang_auto]
            target = [lang_id] + text_ids + [m.eos]

        # trim audio (not text) if the whole thing is too long
        overhead = 2 + 1 + len(control) + len(target)  # bos + audio_end (+audio_start)
        max_audio = max(1, self.max_seq_len - overhead)
        audio = audio[:max_audio]

        prefix = [m.bos, m.audio_start] + audio + [m.audio_end] + control
        input_ids = prefix + target
        labels = [-100] * len(prefix) + target
        return input_ids, labels


class AsrCollator:
    def __init__(self, builder: SequenceBuilder, tmap: TokenMap, p_auto: float, seed: int = 0):
        self.b = builder
        self.m = tmap
        self.p_auto = p_auto
        self.rng = random.Random(seed)

    def __call__(self, rows: list[dict]) -> dict:
        seqs, labs = [], []
        for r in rows:
            mode = "auto" if self.rng.random() < self.p_auto else "forced"
            ids, lab = self.b.build(r["mimi_cb0"], r["text"], r["language"], mode)
            seqs.append(ids)
            labs.append(lab)

        maxlen = max(len(s) for s in seqs)
        pad = self.m.pad
        input_ids, attn, labels = [], [], []
        for ids, lab in zip(seqs, labs):
            n = maxlen - len(ids)
            input_ids.append(ids + [pad] * n)
            attn.append([1] * len(ids) + [0] * n)
            labels.append(lab + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_splits(mimi_dir: str):
    """Return (train_ds, val_ds) from a local encoded dataset dir."""
    from datasets import load_from_disk

    from .load import splits_from_dataset

    return splits_from_dataset(load_from_disk(mimi_dir))
