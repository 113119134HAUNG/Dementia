# -*- coding: utf-8 -*-
"""
config_utils.py

Utility functions for loading and accessing configuration in config_text.yaml.

All path / model / dataset settings are centralized in a single YAML file
to keep experimental code clean and reproducible.

Config structure (expected top-level keys)
-----------------------------------------
- asr        : NCMMSC ASR settings (audio → transcript CSV)
- predictive : Predictive dataset settings (TSV + eGeMAPS)
- text       : Chinese text preprocessing & dataset merging
- features   : Text feature extraction + classifiers + cross-validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Default location of the configuration file
DEFAULT_CONFIG_PATH = Path("config_text.yaml")

# =====================================================================
# Core loader
# =====================================================================
def load_text_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the full configuration dictionary from a YAML file.

    Parameters
    ----------
    path : str or None
        Path to the YAML file. If None, uses DEFAULT_CONFIG_PATH.

    Returns
    -------
    dict
        A dictionary mapping top-level section names to their configs,
        e.g. {"asr": {...}, "predictive": {...}, "text": {...}, "features": {...}}.
    """
    cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cfg_path} 格式錯誤，預期是 dict。")

    return cfg

# =====================================================================
# Section accessors
# =====================================================================
def _get_section(
    section: str,
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Internal helper to fetch a top-level config section."""
    if cfg is None:
        cfg = load_text_config(path)
    if section not in cfg:
        raise KeyError(f"Config 檔缺少 '{section}' 區塊。")
    sub = cfg[section]
    if not isinstance(sub, dict):
        raise ValueError(f"Config 區塊 '{section}' 格式錯誤，預期是 dict。")
    return sub

def get_asr_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'asr' section of the config."""
    return _get_section("asr", cfg=cfg, path=path)

def get_predictive_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'predictive' section of the config."""
    return _get_section("predictive", cfg=cfg, path=path)

def get_text_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'text' section of the config."""
    return _get_section("text", cfg=cfg, path=path)

def get_features_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'features' section of the config."""
    return _get_section("features", cfg=cfg, path=path)
