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
from kupe_asr.hf_utils import ensure_repo, hf_login, log, upload_file


DATA_CARD = """\
# {project} — data

Multilingual ASR corpus for **{project}** (Gemma-3-270m + Mimi codec).

Languages: {langs}

## Configs
- **audio** — raw speech resampled to 24 kHz mono, with normalised transcripts.
- **mimi** — Mimi codebook-0 tokens (12.5 tok/s) + transcripts. Used for training.

```python
from datasets import load_dataset
audio = load_dataset("{data_repo}", "audio", split="train")
mimi  = load_dataset("{data_repo}", "mimi",  split="train")
```

Sources: ai4bharat/IndicVoices, ARTPARK-IISc/Vaani-transcription-part,
mozilla-foundation/common_voice_17_0. See each source for its license.
"""


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
    card = DATA_CARD.format(project=cfg.project, langs=langs, data_repo=cfg.repos.data)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "README.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(card)
        upload_file(p, cfg.repos.data, "dataset", "README.md", "seed data card")

    log.info("repos ready: %s | %s | %s", cfg.repos.data, cfg.repos.runs, cfg.repos.model)


if __name__ == "__main__":
    main()
