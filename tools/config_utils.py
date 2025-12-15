# -*- coding: utf-8 -*-
"""
config_utils.py

Utility functions for loading and accessing configuration in config_text.yaml.

Strict + clean rules:
- Single source of truth: YAML only
- No printing
- No CLI
- Minimal surface area (one loader + one accessor)
- Optional caching to avoid repeated disk reads within a process
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("config_text.yaml")

# =====================================================================
# Core loader
# =====================================================================
def _resolve_path(path: Optional[str]) -> Path:
    p = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    return p

@lru_cache(maxsize=8)
def load_text_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load full YAML config as a dict (cached).

    Notes
    -----
    - Cached by `path` string (None uses DEFAULT_CONFIG_PATH).
    - If you edit the YAML during runtime, call `load_text_config.cache_clear()`.
    """
    cfg_path = _resolve_path(path)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cfg_path} must be a YAML mapping (dict).")

    return cfg

# =====================================================================
# Section accessors
# =====================================================================
def get_section(
    section: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a top-level config section as a dict."""
    root = load_text_config(path) if cfg is None else cfg
    if section not in root:
        raise KeyError(f"Missing top-level config section: {section!r}")
    sub = root[section]
    if not isinstance(sub, dict):
        raise ValueError(f"Config section {section!r} must be a mapping (dict).")
    return sub

def get_asr_config(*, cfg: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Dict[str, Any]:
    return get_section("asr", cfg=cfg, path=path)

def get_predictive_config(*, cfg: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Dict[str, Any]:
    return get_section("predictive", cfg=cfg, path=path)

def get_text_config(*, cfg: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Dict[str, Any]:
    return get_section("text", cfg=cfg, path=path)

def get_features_config(*, cfg: Optional[Dict[str, Any]] = None, path: Optional[str] = None) -> Dict[str, Any]:
    return get_section("features", cfg=cfg, path=path)
