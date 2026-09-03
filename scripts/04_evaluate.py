#!/usr/bin/env python
"""Stage 4: standalone evaluation -> JSON + Markdown report (optionally pushed).

    python scripts/04_evaluate.py --model-dir artifacts/runs/<run>/model
    python scripts/04_evaluate.py --model-dir <dir> --push
"""
import _bootstrap  # noqa: F401
import argparse
import glob
import os

import torch

from kupe_asr.config import load_config
from kupe_asr.data.load import load_mimi_dataset, splits_from_dataset
from kupe_asr.evaluate import run_eval, save_report
from kupe_asr.hf_utils import ensure_repo, hf_login, log, upload_folder
from kupe_asr.tokenizer import load_tokenizer, token_map


def _latest_model_dir(runs_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(runs_dir, "*", "model")))
    if not cands:
        raise SystemExit(f"no run found under {runs_dir}; pass --model-dir")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument(
        "--from-hub", action="store_true",
        help="download the `mimi` config from Hub even if a local copy exists",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_dir = args.model_dir or _latest_model_dir(cfg.paths.runs_dir)
    log.info("evaluating %s", model_dir)

    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(model_dir)
    tmap = token_map(tok)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()

    _, val_ds = splits_from_dataset(load_mimi_dataset(cfg, from_hub=args.from_hub))
    report = run_eval(
        model, tok, tmap, val_ds, device,
        max_audio_frames=cfg.model.max_audio_frames,
        max_samples_per_lang=cfg.eval.max_samples_per_lang,
        auto_samples_per_lang=cfg.eval.auto_samples_per_lang,
        max_new_tokens=cfg.eval.max_new_tokens,
    )
    out_dir = os.path.join(os.path.dirname(model_dir.rstrip("/")), "eval")
    save_report(report, out_dir, title=os.path.basename(os.path.dirname(model_dir.rstrip("/"))))
    print(f"\nWER={report['overall']['wer']:.3f}  CER={report['overall']['cer']:.3f}  "
          f"langID={report['auto']['lang_id_acc']:.3f}\n")

    if args.push:
        hf_login()
        ensure_repo(cfg.repos.runs, "model")
        run_name = os.path.basename(os.path.dirname(model_dir.rstrip("/")))
        upload_folder(out_dir, cfg.repos.runs, "model",
                      path_in_repo=f"runs/{run_name}/eval",
                      commit_message=f"eval {run_name}: WER={report['overall']['wer']:.3f}")


if __name__ == "__main__":
    main()
