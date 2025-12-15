# -*- coding: utf-8 -*-
"""
dataset_subset.py

Subset utilities for Chinese text datasets.

This module is intentionally minimal:
- No CLI
- No file I/O
- No printing (caller decides logging)

Subset rules are driven by `text` config (dict), e.g.:

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
    """Ensure deterministic ordering for reproducible sampling/output."""
    if df is None or df.empty:
        return df
    if "ID" in df.columns:
        return df.sort_values(by="ID", kind="mergesort")
    return df.sort_index(kind="mergesort")

def apply_subset(df: pd.DataFrame, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply subset selection on a unified-schema DataFrame.

    Expected columns (if present):
        - Dataset
        - Diagnosis
        - ID (optional; used for deterministic ordering)

    Config keys (under text_cfg):
        - target_datasets : list[str] | None
        - target_labels   : list[str] | None
        - balance         : bool
        - subset_seed     : int  (required if balance or cap_per_class is used)
        - cap_per_class   : int  | None

    Behavior:
        1) Filter by target_datasets (if provided and column exists)
        2) Filter by target_labels   (if provided and column exists)
        3) If balance=True: downsample each class to min class count
        4) If cap_per_class set: cap each class to at most this size
    """
    if df is None or df.empty:
        return df

    cfg = text_cfg or {}

    target_datasets = _norm_str_list(cfg.get("target_datasets"))
    target_labels = _norm_str_list(cfg.get("target_labels"))

    balance = bool(cfg.get("balance", False))

    cap_raw = cfg.get("cap_per_class", None)
    cap: Optional[int] = int(cap_raw) if cap_raw is not None else None
    if cap is not None and cap < 0:
        raise ValueError("text.cap_per_class must be >= 0 (or null).")

    need_sampling = balance or (cap is not None)
    seed_raw = cfg.get("subset_seed", None)
    if need_sampling and seed_raw is None:
        raise KeyError("text.subset_seed is required when balance or cap_per_class is enabled.")
    subset_seed: Optional[int] = int(seed_raw) if seed_raw is not None else None

    out = df

    # 1) Dataset filter
    if target_datasets is not None and "Dataset" in out.columns:
        keep_ds: Set[str] = set(s.strip() for s in target_datasets if str(s).strip())
        ds = out["Dataset"].astype(str).str.strip()
        out = out[ds.isin(keep_ds)]

    # 2) Label filter
    if target_labels is not None and "Diagnosis" in out.columns:
        keep_lb: Set[str] = {str(x).strip().upper() for x in target_labels if str(x).strip()}
        diag_u = out["Diagnosis"].astype(str).str.strip().str.upper()
        out = out[diag_u.isin(keep_lb)]

    if out.empty:
        return out.reset_index(drop=True)

    # 3/4) Balancing / capping (per Diagnosis)
    if "Diagnosis" not in out.columns:
        return out.reset_index(drop=True)

    diag_u = out["Diagnosis"].astype(str).str.strip().str.upper()
    # drop empty diagnosis strings (avoid grouping on "")
    mask_nonempty = diag_u != ""
    out = out[mask_nonempty]
    diag_u = diag_u[mask_nonempty]

    if out.empty:
        return out.reset_index(drop=True)

    counts = diag_u.value_counts()
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

    assert subset_seed is not None  # guarded by need_sampling check above

    parts: List[pd.DataFrame] = []
    for label in sorted(counts.index.astype(str)):
        grp = out.loc[diag_u == label]

        grp = _stable_sort(grp).reset_index(drop=True)

        take = min(int(n_per), len(grp))
        rs = _stable_label_seed(subset_seed, label)
        parts.append(grp.sample(n=take, random_state=rs, replace=False))

    sampled = pd.concat(parts, axis=0, ignore_index=True)
    sampled = _stable_sort(sampled).reset_index(drop=True)
    return sampled
