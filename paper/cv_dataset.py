# -*- coding: utf-8 -*-
"""
cv_dataset.py

Load pre-cleaned JSONL for evaluation only.
NO preprocessing, NO cleaning, NO splitting here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetView:
    X: List[str]
    y: np.ndarray  # 0/1
    label_names: List[str]  # ["HC","AD"]

def load_cleaned_dataset(cleaned_jsonl: Path) -> DatasetView:
    if not cleaned_jsonl.exists():
        raise FileNotFoundError(f"cleaned JSONL not found: {cleaned_jsonl}")

    df = pd.read_json(cleaned_jsonl, lines=True)

    need_cols = {"Text_interviewer_participant", "Diagnosis"}
    missing = sorted(list(need_cols - set(df.columns)))
    if missing:
        raise ValueError(f"cleaned JSONL missing required columns: {missing}")

    diag = df["Diagnosis"].astype(str).str.strip()

    # paper-style: binary only
    keep = diag.isin(["HC", "AD"])
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        raise ValueError("After filtering to Diagnosis in {HC, AD}, dataset is empty.")

    X = df["Text_interviewer_participant"].fillna("").astype(str).tolist()
    y = df["Diagnosis"].astype(str).str.strip().map({"HC": 0, "AD": 1}).astype(int).to_numpy()

    return DatasetView(X=X, y=y, label_names=["HC", "AD"])
