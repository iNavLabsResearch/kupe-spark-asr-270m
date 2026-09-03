"""Config loading: one YAML file -> nested, dot-accessible object.

`{owner}` in the `repos` section is resolved from the resolved `owner` value,
which itself can be overridden by the HF_OWNER / HF_USERNAME env var.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "config.yaml",
)


class Config(SimpleNamespace):
    """Recursive namespace with dict-style fallback and `.get`."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config(**{k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get("KUPE_CONFIG", DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # owner override precedence: env > yaml
    owner = os.environ.get("HF_OWNER") or os.environ.get("HF_USERNAME") or raw.get("owner")
    raw["owner"] = owner

    # resolve {owner} templates in repo names
    repos = raw.get("repos", {})
    for k, v in list(repos.items()):
        if isinstance(v, str):
            repos[k] = v.format(owner=owner)
    raw["repos"] = repos

    return _wrap(raw)
