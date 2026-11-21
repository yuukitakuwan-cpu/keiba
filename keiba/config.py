"""設定ファイル (config.yaml) の読み込み。"""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

_cache: dict[str, Any] | None = None


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """config.yaml を読み込んで dict を返す。

    環境変数 KEIBA_CONFIG があればそのパスを優先する。
    """
    global _cache
    if path is None and _cache is not None:
        return deepcopy(_cache)
    p = Path(path or os.environ.get("KEIBA_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_base_dir"] = str(p.resolve().parent)
    if path is None:
        _cache = cfg
    return deepcopy(cfg)


def resolve_path(cfg: dict[str, Any], rel: str) -> Path:
    """設定ファイルの場所を基準に相対パスを解決する。"""
    p = Path(rel)
    if p.is_absolute():
        return p
    return Path(cfg.get("_base_dir", ROOT)) / p


def database_url(cfg: dict[str, Any]) -> str:
    url = cfg["database"]["url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url[len("sqlite:///"):]
        if rel != ":memory:":
            p = resolve_path(cfg, rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{p}"
    return url
