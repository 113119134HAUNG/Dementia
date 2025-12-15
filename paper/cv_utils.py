# -*- coding: utf-8 -*-
"""
cv_utils.py

Small strict helpers shared by evaluate_cv modules.
No preprocessing logic. No modeling logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

def require(cfg: Dict[str, Any], key: str, *, where: str = "") -> Any:
    if key not in cfg:
        prefix = f"{where}." if where else ""
        raise KeyError(f"Config missing required key: {prefix}{key}")
    return cfg[key]

def get_dict(cfg: Dict[str, Any], key: str, *, where: str = "") -> Dict[str, Any]:
    v = require(cfg, key, where=where)
    if not isinstance(v, dict):
        prefix = f"{where}." if where else ""
        raise ValueError(f"{prefix}{key} must be a dict.")
    return v

def norm_str_list(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out = [str(i).strip() for i in x if str(i).strip()]
        return out or None
    s = str(x).strip()
    return [s] if s else None

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
