"""Environment, HF auth, repo creation/push, wandb init, logging helpers.

Secrets come from a .env file (auto-loaded) OR exported env vars:
    HF_TOKEN, WANDB_API_KEY, WANDB_PROJECT, HF_OWNER
Nothing here fails hard if a secret is missing — callers decide what is required.
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("kupe")

# keep the terminal readable: silence per-request HTTP chatter so tqdm shows.
for _n in ("httpx", "urllib3", "filelock", "fsspec", "huggingface_hub",
           "huggingface_hub.hf_api", "datasets"):
    logging.getLogger(_n).setLevel(logging.WARNING)

_ENV_LOADED = False


def load_env() -> None:
    """Load .env once (if python-dotenv is available). Idempotent."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        # search upward from CWD for a .env
        load_dotenv(override=False)
    except Exception:  # dotenv optional; exported vars still work
        pass
    _ENV_LOADED = True


def hf_token() -> str | None:
    load_env()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def require_token() -> str:
    tok = hf_token()
    if not tok:
        raise RuntimeError(
            "HF_TOKEN not set. Put it in .env or run:  export HF_TOKEN=hf_xxx"
        )
    return tok


def hf_login() -> str:
    """Log the process into the Hub and return the token."""
    from huggingface_hub import login

    tok = require_token()
    login(token=tok, add_to_git_credential=False)
    return tok


def ensure_repo(repo_id: str, repo_type: str, private: bool = False) -> str:
    """Create a Hub repo if missing; return the repo_id."""
    from huggingface_hub import create_repo

    create_repo(
        repo_id,
        repo_type=repo_type,
        token=require_token(),
        private=private,
        exist_ok=True,
    )
    log.info("repo ready: %s (%s)", repo_id, repo_type)
    return repo_id


def upload_folder(folder: str, repo_id: str, repo_type: str, path_in_repo: str = "",
                  commit_message: str = "upload", ignore_patterns: list[str] | None = None) -> None:
    from huggingface_hub import upload_folder as _upload

    _upload(
        folder_path=folder,
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=path_in_repo,
        token=require_token(),
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    log.info("pushed %s -> %s:%s", folder, repo_id, path_in_repo or "/")


def upload_file(local_path: str, repo_id: str, repo_type: str, path_in_repo: str,
                commit_message: str = "upload") -> None:
    from huggingface_hub import upload_file as _upload

    _upload(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=require_token(),
        commit_message=commit_message,
    )


def init_wandb(project_default: str, run_name: str, config: dict | None = None):
    """Init wandb if a key is present; otherwise return None (training still runs)."""
    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        log.warning("WANDB_API_KEY not set -> wandb disabled (metrics still print).")
        os.environ["WANDB_DISABLED"] = "true"
        return None
    try:
        import wandb

        project = os.environ.get("WANDB_PROJECT", project_default)
        return wandb.init(project=project, name=run_name, config=config or {})
    except Exception as e:  # never let telemetry kill a run
        log.warning("wandb init failed (%s) -> continuing without it.", e)
        os.environ["WANDB_DISABLED"] = "true"
        return None
