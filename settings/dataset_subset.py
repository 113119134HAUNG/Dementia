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


# -----------------------------
# small helpers (pure)
# -----------------------------
def _norm_str_list(xs: Any) -> Optional[List[str]]:
    """Normalize a config entry into a list[str] (or None)."""
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


def _parse_optional_int(cfg: Dict[str, Any], key: str) -> Optional[int]:
    """Parse optional int config; return None if missing/null, else int."""
    if key not in cfg or cfg.get(key) is None:
        return None
    v = cfg.get(key)

    # Reject bool explicitly (bool is subclass of int)
    if isinstance(v, bool):
        raise ValueError(f"text.{key} must be an int (or null), got bool.")

    try:
        iv = int(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"text.{key} must be an int (or null), got: {v!r}") from e
    return iv


def _parse_bool_strict(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    """
    Strict bool parser:
    - accept only True/False (actual bool type)
    - if missing -> default
    """
    if key not in cfg or cfg.get(key) is None:
        return bool(default)
    v = cfg.get(key)
    if not isinstance(v, bool):
        raise ValueError(f"text.{key} must be a bool (true/false), got: {type(v).__name__}")
    return bool(v)


def _stable_label_seed(base_seed: int, label: str) -> int:
    """Stable label-specific seed (do NOT use Python's hash())."""
    h = zlib.crc32(label.encode("utf-8")) % 100000
    return int(base_seed) + int(h)


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure deterministic ordering for reproducible sampling/output."""
    if df is None or df.empty:
        return df

    # Prefer stable multi-key ordering if available
    cols: List[str] = []
    if "Dataset" in df.columns:
        cols.append("Dataset")
    if "ID" in df.columns:
        cols.append("ID")

    if cols:
        return df.sort_values(by=cols, kind="mergesort")

    # Fallback to ID-only, then index
    if "ID" in df.columns:
        return df.sort_values(by="ID", kind="mergesort")

    return df.sort_index(kind="mergesort")


def _upper_strip_series(s: pd.Series) -> pd.Series:
    """Upper+strip for robust label comparisons."""
    return s.astype(str).str.strip().str.upper()


# -----------------------------
# main API
# -----------------------------
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
    """
    if df is None or df.empty:
        return df

    cfg = text_cfg or {}

    target_datasets = _norm_str_list(cfg.get("target_datasets"))
    target_labels = _norm_str_list(cfg.get("target_labels"))

    balance = _parse_bool_strict(cfg, "balance", default=False)

    cap = _parse_optional_int(cfg, "cap_per_class")
    if cap is not None and cap < 0:
        raise ValueError("text.cap_per_class must be >= 0 (or null).")

    need_sampling = balance or (cap is not None)

    subset_seed = _parse_optional_int(cfg, "subset_seed")
    if need_sampling and subset_seed is None:
        raise KeyError("text.subset_seed is required when balance or cap_per_class is enabled.")
    if subset_seed is not None and subset_seed < 0:
        raise ValueError("text.subset_seed must be >= 0.")

    out = df

    # 1) Dataset filter
    if target_datasets is not None and "Dataset" in out.columns:
        keep_ds: Set[str] = {str(s).strip() for s in target_datasets if str(s).strip()}
        ds = out["Dataset"].astype(str).str.strip()
        out = out[ds.isin(keep_ds)]

    # 2) Label filter
    if target_labels is not None and "Diagnosis" in out.columns:
        keep_lb: Set[str] = {str(x).strip().upper() for x in target_labels if str(x).strip()}
        diag_u = _upper_strip_series(out["Diagnosis"])
        out = out[diag_u.isin(keep_lb)]

    if out.empty:
        return out.reset_index(drop=True)

    # 3/4) Balancing / capping (per Diagnosis)
    if "Diagnosis" not in out.columns:
        return out.reset_index(drop=True)

    diag_u = _upper_strip_series(out["Diagnosis"])

    # drop empty diagnosis strings
    nonempty = diag_u != ""
    out = out[nonempty]
    diag_u = diag_u[nonempty]

    if out.empty:
        return out.reset_index(drop=True)

    counts = diag_u.value_counts()
    if counts.empty:
        return out.reset_index(drop=True)

    # compute n_per
    n_per: Optional[int] = None
    if balance:
        n_per = int(counts.min())

    if cap is not None:
        n_per = cap if n_per is None else min(n_per, cap)

    # No sampling requested
    if n_per is None:
        return out.reset_index(drop=True)

    # Cap to zero means empty output
    if n_per <= 0:
        return out.iloc[0:0].reset_index(drop=True)

    assert subset_seed is not None  # guarded above

    # sample per label deterministically
    unique_labels = sorted(set(counts.index.astype(str).tolist()))
    parts: List[pd.DataFrame] = []

    for label in unique_labels:
        grp = out.loc[diag_u == label]
        grp = _stable_sort(grp).reset_index(drop=True)

        take = min(int(n_per), len(grp))
        rs = _stable_label_seed(int(subset_seed), label)

        # pandas sample is deterministic given stable order + random_state
        parts.append(grp.sample(n=take, random_state=rs, replace=False))

    sampled = pd.concat(parts, axis=0, ignore_index=True)
    sampled = _stable_sort(sampled).reset_index(drop=True)
    return sampled