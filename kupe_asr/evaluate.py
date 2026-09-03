"""Evaluation: forced-mode WER/CER per language + auto-mode language-ID accuracy.

Produces a JSON + Markdown report. Used both as a training callback (small subset)
and as a standalone end-of-run evaluation that is pushed to the runs repo.
"""
from __future__ import annotations

import json
import os

import torch
from tqdm import tqdm

from .constants import LANGUAGES
from .hf_utils import log
from .text import normalize
from .tokenizer import TokenMap


# ------------------------------------------------------------------ generation
def _prefix(tmap: TokenMap, mimi_cb0, lang: str, mode: str, max_audio_frames: int):
    audio = [tmap.mimi_base + int(c) for c in mimi_cb0[:max_audio_frames]]
    tail = tmap.lang_auto if mode == "auto" else tmap.lang[lang]
    return [tmap.bos, tmap.audio_start] + audio + [tmap.audio_end, tail]


@torch.inference_mode()
def _generate(model, tmap: TokenMap, prefixes, device, max_new_tokens, batch_size=16):
    """Left-pad batched greedy generation; return list of generated-id lists."""
    outs = []
    for i in tqdm(range(0, len(prefixes), batch_size), desc="generate", leave=False):
        chunk = prefixes[i : i + batch_size]
        maxlen = max(len(p) for p in chunk)
        input_ids, attn = [], []
        for p in chunk:
            n = maxlen - len(p)
            input_ids.append([tmap.pad] * n + p)   # LEFT pad for decoder-only gen
            attn.append([0] * n + [1] * len(p))
        input_ids = torch.tensor(input_ids, device=device)
        attn = torch.tensor(attn, device=device)
        gen = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            eos_token_id=tmap.eos, pad_token_id=tmap.pad,
        )
        for j in range(len(chunk)):
            outs.append(gen[j, maxlen:].tolist())  # strip the (padded) prefix
    return outs


def _decode_text(tokenizer, ids, tmap: TokenMap):
    if tmap.eos in ids:
        ids = ids[: ids.index(tmap.eos)]
    return normalize(tokenizer.decode(ids, skip_special_tokens=True))


# ------------------------------------------------------------------ scoring
def _wer_cer(refs, hyps):
    import jiwer

    if not refs:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    return {
        "wer": float(jiwer.wer(refs, hyps)),
        "cer": float(jiwer.cer(refs, hyps)),
        "n": len(refs),
    }


def run_eval(model, tokenizer, tmap: TokenMap, val_ds, device, *,
             max_audio_frames: int, max_samples_per_lang: int,
             auto_samples_per_lang: int, max_new_tokens: int) -> dict:
    model.eval()
    report = {"per_language": {}, "auto": {"per_language": {}}, "samples": []}
    all_refs, all_hyps = [], []

    for lang in LANGUAGES:
        sub = val_ds.filter(lambda s, L=lang: s == L, input_columns="language")
        if sub.num_rows == 0:
            continue
        sub = sub.select(range(min(max_samples_per_lang, sub.num_rows)))
        refs = [normalize(t) for t in sub["text"]]
        prefixes = [_prefix(tmap, c, lang, "forced", max_audio_frames) for c in sub["mimi_cb0"]]
        gens = _generate(model, tmap, prefixes, device, max_new_tokens)
        hyps = [_decode_text(tokenizer, g, tmap) for g in gens]
        report["per_language"][lang] = _wer_cer(refs, hyps)
        all_refs += refs
        all_hyps += hyps
        for r, h in list(zip(refs, hyps))[:2]:
            report["samples"].append({"lang": lang, "ref": r, "hyp": h})

    report["overall"] = _wer_cer(all_refs, all_hyps)

    # ---- auto mode: language id accuracy + WER ----
    a_refs, a_hyps, correct, total = [], [], 0, 0
    for lang in LANGUAGES:
        sub = val_ds.filter(lambda s, L=lang: s == L, input_columns="language")
        if sub.num_rows == 0:
            continue
        sub = sub.select(range(min(auto_samples_per_lang, sub.num_rows)))
        refs = [normalize(t) for t in sub["text"]]
        prefixes = [_prefix(tmap, c, lang, "auto", max_audio_frames) for c in sub["mimi_cb0"]]
        gens = _generate(model, tmap, prefixes, device, max_new_tokens)
        n_ok = 0
        hyps = []
        for g in gens:
            pred_code = tmap.lang_id_to_code.get(g[0]) if g else None
            n_ok += int(pred_code == lang)
            rest = g[1:] if (g and g[0] in tmap.lang_id_to_code) else g
            hyps.append(_decode_text(tokenizer, rest, tmap))
        acc = n_ok / max(1, len(gens))
        report["auto"]["per_language"][lang] = {"lang_id_acc": acc, **_wer_cer(refs, hyps)}
        correct += n_ok
        total += len(gens)
        a_refs += refs
        a_hyps += hyps
    report["auto"]["lang_id_acc"] = correct / max(1, total)
    report["auto"].update(_wer_cer(a_refs, a_hyps))
    return report


# ------------------------------------------------------------------ reporting
def render_markdown(report: dict, title: str = "kupe-spark-asr-270m eval") -> str:
    o = report["overall"]
    lines = [f"# {title}", "",
             f"**Overall (forced):** WER `{o['wer']:.3f}`  CER `{o['cer']:.3f}`  (n={o['n']})",
             f"**Auto lang-ID accuracy:** `{report['auto']['lang_id_acc']:.3f}`  "
             f"(auto WER `{report['auto'].get('wer', float('nan')):.3f}`)", "",
             "## Forced mode (per language)", "",
             "| lang | name | WER | CER | n |", "|---|---|---|---|---|"]
    for code, name in LANGUAGES.items():
        r = report["per_language"].get(code)
        if r:
            lines.append(f"| {code} | {name} | {r['wer']:.3f} | {r['cer']:.3f} | {r['n']} |")
    lines += ["", "## Auto mode (per language)", "",
              "| lang | lang-ID acc | WER | CER | n |", "|---|---|---|---|---|"]
    for code in LANGUAGES:
        r = report["auto"]["per_language"].get(code)
        if r:
            lines.append(f"| {code} | {r['lang_id_acc']:.3f} | {r['wer']:.3f} | {r['cer']:.3f} | {r['n']} |")
    lines += ["", "## Samples", ""]
    for s in report["samples"][:12]:
        lines.append(f"- **[{s['lang']}]** ref: `{s['ref']}`  →  hyp: `{s['hyp']}`")
    return "\n".join(lines) + "\n"


def save_report(report: dict, out_dir: str, title: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "eval_report.json")
    mpath = os.path.join(out_dir, "eval_report.md")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(render_markdown(report, title))
    log.info("eval report -> %s", out_dir)
    return jpath, mpath
