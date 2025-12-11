"""
config_utils.py
Utility functions for loading and accessing configuration in config_text.yaml.

This module keeps all path / model / dataset settings in a single YAML
file, so that experimental code can remain clean and reproducible.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Default location of the text/ASR configuration file
DEFAULT_CONFIG_PATH = Path("config_text.yaml")

def load_text_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the full configuration dictionary from a YAML file.

    Args:
        path: Optional path to the YAML file. If None, uses DEFAULT_CONFIG_PATH.

    Returns:
        A dictionary with (at least) the keys "asr" and "text".
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} 內容格式錯誤，預期是 dict。")

    return cfg

def get_asr_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'asr' section of the config.

    You can either pass an already-loaded ``cfg`` or let this function
    load from ``path`` / DEFAULT_CONFIG_PATH.
    """
    if cfg is None:
        cfg = load_text_config(path)
    if "asr" not in cfg:
        raise KeyError("Config 檔缺少 'asr' 區塊。")
    return cfg["asr"]

def get_text_config(
    cfg: Optional[Dict[str, Any]] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the 'text' section of the config."""
    if cfg is None:
        cfg = load_text_config(path)
    if "text" not in cfg:
        raise KeyError("Config 檔缺少 'text' 區塊。")
    return cfg["text"]
