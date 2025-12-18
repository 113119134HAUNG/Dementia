# -*- coding: utf-8 -*-
"""
dataset_subset.py

Subset utilities for Chinese text datasets.

This module is intentionally minimal:
- No CLI
- No file I/O
- No printing (caller decides logging)
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


def _parse_optional_int(cfg: Dict[str, Any], key: str) -> Optional[int]:
    if key not in cfg or cfg.get(key) is None:
        return None
    v = cfg.get(key)
    if isinstance(v, bool):
        raise ValueError(f"text.{key} must be an int (or null), got bool.")
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"text.{key} must be an int (or null), got: {v!r}") from e


def _stable_label_seed(base_seed: int, label: str) -> int:
    h = zlib.crc32(label.encode("utf-8")) % 100000
    return int(base_seed) + int(h)


def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    cols: List[str] = []
    if "Dataset" in df.columns:
        cols.append("Dataset")
    if "ID" in df.columns:
        cols.append("ID")

    if cols:
        return df.sort_values(by=cols, kind="mergesort")

    return df.sort_index(kind="mergesort")


def apply_subset(df: pd.DataFrame, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    cfg = text_cfg or {}

    target_datasets = _norm_str_list(cfg.get("target_datasets"))
    target_labels = _norm_str_list(cfg.get("target_labels"))

    balance_raw = cfg.get("balance", False)
    if isinstance(balance_raw, (int, str)) and not isinstance(balance_raw, bool):
        raise ValueError("text.balance must be a bool.")
    balance = bool(balance_raw)

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

    if target_datasets is not None and "Dataset" in out.columns:
        keep_ds: Set[str] = {str(s).strip() for s in target_datasets if str(s).strip()}
        ds = out["Dataset"].astype(str).str.strip()
        out = out[ds.isin(keep_ds)]

    if target_labels is not None and "Diagnosis" in out.columns:
        keep_lb: Set[str] = {str(x).strip().upper() for x in target_labels if str(x).strip()}
        diag_u = out["Diagnosis"].astype(str).str.strip().str.upper()
        out = out[diag_u.isin(keep_lb)]

    if out.empty:
        return out.reset_index(drop=True)

    if "Diagnosis" not in out.columns:
        return out.reset_index(drop=True)

    diag_u = out["Diagnosis"].astype(str).str.strip().str.upper()
    out = out[diag_u != ""]
    diag_u = diag_u[diag_u != ""]

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

    assert subset_seed is not None

    parts: List[pd.DataFrame] = []
    for label in sorted(counts.index.astype(str)):
        grp = out.loc[diag_u == label]
        grp = _stable_sort(grp).reset_index(drop=True)

        take = min(int(n_per), len(grp))
        rs = _stable_label_seed(int(subset_seed), label)
        parts.append(grp.sample(n=take, random_state=rs, replace=False))

    sampled = pd.concat(parts, axis=0, ignore_index=True)
    sampled = _stable_sort(sampled).reset_index(drop=True)
    return sampled