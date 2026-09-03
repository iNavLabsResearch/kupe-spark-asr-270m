#!/usr/bin/env python
"""Create the Hub repos (data + runs + model) and seed the data card.

    python scripts/00_create_repos.py
"""
import _bootstrap  # noqa: F401
import argparse
import os
import tempfile

from kupe_asr.config import load_config
from kupe_asr.constants import LANGUAGES
from kupe_asr.data.fetch_state import audio_data_card
from kupe_asr.hf_utils import ensure_repo, hf_login, log, upload_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login()

    ensure_repo(cfg.repos.data, "dataset", private=args.private)
    ensure_repo(cfg.repos.runs, "model", private=args.private)
    ensure_repo(cfg.repos.model, "model", private=args.private)

    langs = ", ".join(f"{c} ({n})" for c, n in LANGUAGES.items())
    card = audio_data_card(cfg.project, langs, cfg.repos.data)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "README.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(card)
        upload_file(p, cfg.repos.data, "dataset", "README.md", "seed data card")

    log.info("repos ready: %s | %s | %s", cfg.repos.data, cfg.repos.runs, cfg.repos.model)


if __name__ == "__main__":
    main()
