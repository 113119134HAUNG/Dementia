# -*- coding: utf-8 -*-
"""
config_utils.py

Utility functions for loading and accessing configuration in config_text.yaml.

Strict + clean rules:
- Single source of truth: YAML only
- No printing
- No CLI
- Minimal surface area (one loader + section accessors)
- Cached to avoid repeated disk reads within a process

Paper-strict note
-----------------
We normalize the config path for caching so that equivalent paths
(e.g., "./config_text.yaml" vs "config_text.yaml") share the same cache entry.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("config_text.yaml")

# =====================================================================
# Core path resolver
# =====================================================================
def _resolve_path(path: Optional[str]) -> Path:
    p = DEFAULT_CONFIG_PATH if path is None else Path(path)
    p = p.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    return p

def _cache_key(path: Optional[str]) -> str:
    """Canonical cache key for a config path."""
    return str(_resolve_path(path))

# =====================================================================
# Core loader (cached)
# =====================================================================
@lru_cache(maxsize=8)
def _load_text_config_by_key(path_key: str) -> Dict[str, Any]:
    cfg_path = Path(path_key)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cfg_path} must be a YAML mapping (dict).")
    return cfg

def load_text_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load full YAML config as a dict (cached).

    Notes
    -----
    - Cached by normalized absolute path string.
    - If you edit the YAML during runtime, call `clear_config_cache()`.
    """
    return _load_text_config_by_key(_cache_key(path))

def clear_config_cache() -> None:
    """Clear YAML config cache (useful in notebooks after editing YAML)."""
    _load_text_config_by_key.cache_clear()

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
