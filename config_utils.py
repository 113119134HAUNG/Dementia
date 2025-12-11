# config_utils.py
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path("config_text.yaml")

# 讀取設定檔，回傳dict，包含 "asr" 與 "text" 兩個 section。
def load_text_config(path: Optional[str] = None) -> Dict[str, Any]:

    if path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} 內容格式錯誤，預期是 dict。")

    return cfg

# 取得 asr 區塊設定。
def get_asr_config(cfg: Optional[Dict[str, Any]] = None,
                   path: Optional[str] = None) -> Dict[str, Any]:

    if cfg is None:
        cfg = load_text_config(path)
    if "asr" not in cfg:
        raise KeyError("Config 檔缺少 'asr' 區塊。")
    return cfg["asr"]

# 取得 text 區塊設定
def get_text_config(cfg: Optional[Dict[str, Any]] = None,
                    path: Optional[str] = None) -> Dict[str, Any]:

    if cfg is None:
        cfg = load_text_config(path)
    if "text" not in cfg:
        raise KeyError("Config 檔缺少 'text' 區塊。")
    return cfg["text"]
