"""Load Gemma-3-270m as a causal LM and grow its embeddings for our new tokens.

New rows (control + 2048 audio tokens) are initialised to the mean of the existing
embedding matrix — a standard, stable init that starts the audio tokens near the
text manifold instead of at random noise.
"""
from __future__ import annotations

import torch

from .hf_utils import hf_token, log


def _from_pretrained(cls, model_id, dtype, attn_impl):
    """from_pretrained across transformers versions (dtype vs torch_dtype kwarg)."""
    common = dict(token=hf_token(), attn_implementation=attn_impl)
    try:
        return cls.from_pretrained(model_id, dtype=dtype, **common)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=dtype, **common)


def load_model(base_id: str, new_vocab_size: int, dtype: str = "bfloat16",
               attn_impl: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    model = _from_pretrained(AutoModelForCausalLM, base_id, torch_dtype, attn_impl)

    old_vocab = model.get_input_embeddings().weight.shape[0]
    if new_vocab_size == old_vocab:
        log.info("vocab unchanged (%d)", old_vocab)
        return model

    # mean-init the appended rows
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean_vec = emb[:old_vocab].mean(dim=0, keepdim=True)
        model.resize_token_embeddings(new_vocab_size)
        new_emb = model.get_input_embeddings().weight
        new_emb[old_vocab:] = mean_vec
        # untied lm_head (Gemma ties, so this is usually a no-op safety net)
        out = model.get_output_embeddings()
        if out is not None and out.weight.data_ptr() != new_emb.data_ptr():
            out.weight[old_vocab:] = mean_vec

    log.info("resized embeddings %d -> %d (mean-init new rows)", old_vocab, new_vocab_size)
    return model
