# -*- coding: utf-8 -*-
"""
cv_folds.py

Create / save / load StratifiedKFold indices once (paper-style reuse).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cv_utils import ensure_parent

def build_folds_indices(
    y: np.ndarray,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 1:
        raise ValueError("features.crossval.n_splits must be >= 2")

    skf = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=bool(shuffle),
        random_state=int(random_state) if shuffle else None,
    )
    idx = np.arange(len(y))
    return list(skf.split(idx, y))


def save_folds_indices(
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    payload = [
        {"train_idx": tr.astype(int).tolist(), "test_idx": te.astype(int).tolist()}
        for tr, te in folds
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_folds_indices(path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"fold indices file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("fold indices JSON must be a list.")

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict) or "train_idx" not in item or "test_idx" not in item:
            raise ValueError(f"Bad fold item at index {i}: expected dict with train_idx/test_idx.")
        tr = np.array(item["train_idx"], dtype=int)
        te = np.array(item["test_idx"], dtype=int)
        folds.append((tr, te))
    return folds
