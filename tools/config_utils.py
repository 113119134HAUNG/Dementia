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
    if path is None:
        cfg_path = DEFAULT_CONFIG_PATH
    else:
        cfg_path = Path(path)

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cfg_path} 格式錯誤，預期是 dict。")

    return cfg

# =====================================================================
# Section accessors
# =====================================================================

def get_asr_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'asr' section of the config.

    Parameters
    ----------
    cfg : dict or None
        Already loaded config dictionary. If None, the YAML will be
        loaded from ``path`` / DEFAULT_CONFIG_PATH.
    path : str or None
        Path to the YAML file (used only when cfg is None).

    Raises
    ------
    KeyError
        If the 'asr' section is missing.
    """
    if cfg is None:
        cfg = load_text_config(path)
    if "asr" not in cfg:
        raise KeyError("Config 檔缺少 'asr' 區塊。")
    return cfg["asr"]

def get_predictive_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'predictive' section of the config.

    This section is intended for the predictive challenge dataset:
        - meta CSV (uuid, label, demographics)
        - eGeMAPS CSV
        - TSV root directory
        - output paths for text JSONL & eGeMAPS feature CSV

    Parameters
    ----------
    cfg : dict or None
        Already loaded config dictionary. If None, the YAML will be
        loaded from ``path`` / DEFAULT_CONFIG_PATH.
    path : str or None
        Path to the YAML file (used only when cfg is None).

    Raises
    ------
    KeyError
        If the 'predictive' section is missing.
    """
    if cfg is None:
        cfg = load_text_config(path)
    if "predictive" not in cfg:
        raise KeyError("Config 檔缺少 'predictive' 區塊。")
    return cfg["predictive"]

def get_text_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'text' section of the config.

    This section controls Chinese text preprocessing & dataset merging:
        - paths for NCMMSC JSONL / predictive JSONL / optional TAUKADIAL JSONL
        - output directory and filenames for merged / train / test JSONL
    """
    if cfg is None:
        cfg = load_text_config(path)
    if "text" not in cfg:
        raise KeyError("Config 檔缺少 'text' 區塊。")
    return cfg["text"]

def get_features_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'features' section of the config.

    This section controls:
        - which text feature method to run (tfidf / bert / glove / gemma)
        - per-method hyper-parameters
        - cross-validation settings
    """
    if cfg is None:
        cfg = load_text_config(path)
    if "features" not in cfg:
        raise KeyError("Config 檔缺少 'features' 區塊。")
    return cfg["features"]
