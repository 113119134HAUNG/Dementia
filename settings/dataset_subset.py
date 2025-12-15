# -*- coding: utf-8 -*-
"""
dataset_subset.py

Subset utilities for Chinese text datasets.

This module is intentionally minimal:
- No CLI
- No file I/O
- No printing (caller decides logging)

It applies subset rules driven by `text` config (dict), e.g.:

text:
  target_datasets: ["NCMMSC2021_AD_Competition"]
  target_labels: ["AD", "HC"]
  balance: true
  subset_seed: 42
  cap_per_class: 300
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pandas as pd
import zlib


def _norm_str_list(xs: Any) -> Optional[List[str]]:
    if xs is None:
        return None
    if isinstance(xs, (list, tuple)):
        out: List[str] = []
        for x in xs:
            s = str(x).strip()
            if s:
                out.append(s)
        return out if out else None
    s = str(xs).strip()
    return [s] if s else None

def _stable_label_seed(base_seed: int, label: str) -> int:
    # stable across runs (do NOT use Python's hash())
    h = zlib.crc32(label.encode("utf-8")) % 100000
    return int(base_seed) + int(h)

def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure deterministic ordering before sampling."""
    if df is None or df.empty:
        return df
    if "ID" in df.columns:
        # stable sort on ID for reproducible sampling
        return df.sort_values(by="ID", kind="mergesort")
    # fallback: stable sort on index
    return df.sort_index(kind="mergesort")

def apply_subset(df: pd.DataFrame, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply subset selection on a unified-schema DataFrame.

    Expected columns (if present):
        - Dataset
        - Diagnosis
        - ID (optional; used only for stable sampling order)

    Config keys (under text_cfg):
        - target_datasets : list[str] | None
        - target_labels   : list[str] | None
        - balance         : bool (default False)
        - subset_seed     : int  (default 42)
        - cap_per_class   : int  | None

    Behavior:
        1) Filter by target_datasets (if provided and column exists)
        2) Filter by target_labels   (if provided and column exists)
        3) If balance=True: downsample each class to the same size (= min class count)
        4) If cap_per_class is set: additionally cap each class to at most this size
    """
    if df is None or df.empty:
        return df

    cfg = text_cfg or {}

    target_datasets = _norm_str_list(cfg.get("target_datasets"))
    target_labels = _norm_str_list(cfg.get("target_labels"))

    balance = bool(cfg.get("balance", False))
    subset_seed = int(cfg.get("subset_seed", 42))

    cap_per_class = cfg.get("cap_per_class", None)
    cap: Optional[int] = int(cap_per_class) if cap_per_class is not None else None

    out = df

    #  Dataset filter
    if target_datasets is not None and "Dataset" in out.columns:
        keep_ds: Set[str] = set(target_datasets)
        out = out[out["Dataset"].astype(str).isin(keep_ds)]

    #  Label filter
    if target_labels is not None and "Diagnosis" in out.columns:
        keep_lb: Set[str] = {str(x).strip().upper() for x in target_labels if str(x).strip()}
        out = out[out["Diagnosis"].astype(str).str.upper().isin(keep_lb)]

    if out.empty:
        return out.reset_index(drop=True)

    #  Balancing / capping (per Diagnosis)
    if "Diagnosis" not in out.columns:
        return out.reset_index(drop=True)

    counts = out["Diagnosis"].astype(str).value_counts()
    if counts.empty:
        return out.reset_index(drop=True)

    n_per: Optional[int] = None
    if balance:
        n_per = int(counts.min())

    if cap is not None:
        n_per = cap if n_per is None else min(n_per, cap)

    if n_per is None:
        return out.reset_index(drop=True)

    if n_per <= 0:
        return out.iloc[0:0].reset_index(drop=True)

    parts: List[pd.DataFrame] = []
    for label in sorted(counts.index.astype(str)):
        grp = out[out["Diagnosis"].astype(str) == label]

        # stable ordering before sampling (important for reproducibility)
        grp = _stable_sort(grp).reset_index(drop=True)

        take = min(n_per, len(grp))
        rs = _stable_label_seed(subset_seed, label)
        parts.append(grp.sample(n=take, random_state=rs, replace=False))

    return pd.concat(parts, axis=0, ignore_index=True).reset_index(drop=True)
