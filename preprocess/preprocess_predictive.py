# -*- coding: utf-8 -*-
"""
preprocess_predictive.py (paper-strict, converged)

- Load YAML once.
- TSV parsing behavior is driven by predictive.tsv:
    keep_speakers / drop_silence / order_by
- dataset_name is driven by predictive.dataset_name
- paper-strict:
    * if order_by is set, required TSV columns MUST exist (fail-fast, no silent fallback)
    * deterministic output order via sorting by uuid after dedup
    * keep_speakers supports backward-compat: predictive.keep_speakers (fallback)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from settings.enums import ADType
from tools.config_utils import load_text_config, get_predictive_config

# =====================================================================
# Strict helpers
# =====================================================================
def _require(cfg: Dict[str, Any], key: str, *, where: str = "") -> Any:
    if key not in cfg:
        prefix = f"{where}." if where else ""
        raise KeyError(f"Config missing required key: {prefix}{key}")
    return cfg[key]

def _get_dict(cfg: Dict[str, Any], key: str, *, where: str = "") -> Dict[str, Any]:
    v = _require(cfg, key, where=where)
    if not isinstance(v, dict):
        prefix = f"{where}." if where else ""
        raise ValueError(f"{prefix}{key} must be a dict.")
    return v

def _norm_str_list(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out = [str(i).strip() for i in x if str(i).strip()]
        return out or None
    s = str(x).strip()
    return [s] if s else None

# =====================================================================
# uuid normalize + dedup (single point)
# =====================================================================
_INVALID_UUID = {"", "nan", "none", "null"}

def _normalize_and_dedup_uuid(df: pd.DataFrame, *, name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df is None or df.empty or "uuid" not in df.columns:
        return df, {"dropped_invalid": 0, "dropped_dups": 0}

    out = df.copy()
    uu = out["uuid"].astype(str).str.strip()
    uu_low = uu.str.lower()

    valid = ~uu_low.isin(_INVALID_UUID)
    dropped_invalid = int((~valid).sum())
    out = out.loc[valid].copy()
    out["uuid"] = uu.loc[valid].astype(str).str.strip()

    dup_mask = out["uuid"].duplicated(keep="first")
    dropped_dups = int(dup_mask.sum())
    if dropped_dups > 0:
        out = out.loc[~dup_mask].copy()

    return out.reset_index(drop=True), {"dropped_invalid": dropped_invalid, "dropped_dups": dropped_dups}

# =====================================================================
# TSV -> long-form text (YAML-driven, fail-fast)
# =====================================================================
def tsv_to_text(
    tsv_path: Path,
    *,
    keep_speakers: Optional[List[str]],
    drop_silence: bool,
    order_by: Optional[str],
) -> str:
    df = pd.read_csv(tsv_path, sep="\t", keep_default_na=False)

    if "value" not in df.columns:
        raise ValueError(f"TSV file {tsv_path} has no 'value' column.")

    if keep_speakers is not None:
        if "speaker" not in df.columns:
            raise ValueError(f"TSV file {tsv_path} has no 'speaker' column.")
        ks = {str(x).strip() for x in keep_speakers if str(x).strip()}
        df = df[df["speaker"].astype(str).str.strip().isin(ks)]

    # paper-strict: if order_by is specified, required columns MUST exist (no silent fallback)
    if order_by == "no":
        if "no" not in df.columns:
            raise ValueError(f"TSV file {tsv_path} missing required column 'no' for order_by='no'.")
        df["_no"] = pd.to_numeric(df["no"], errors="coerce")
        df = df.sort_values(by="_no", kind="mergesort").drop(columns=["_no"])
    elif order_by == "start_time":
        if "start_time" not in df.columns:
            raise ValueError(f"TSV file {tsv_path} missing required column 'start_time' for order_by='start_time'.")
        df["_st"] = pd.to_numeric(df["start_time"], errors="coerce")
        df = df.sort_values(by="_st", kind="mergesort").drop(columns=["_st"])

    vals = df["value"].astype(str).str.strip()
    vals = vals[vals != ""]

    if drop_silence:
        vals = vals[vals.str.lower() != "sil"]

    return " ".join(vals.tolist())

# =====================================================================
# Build outputs
# =====================================================================
def build_text_jsonl(
    meta_df: pd.DataFrame,
    *,
    tsv_root: Path,
    output_jsonl: Path,
    dataset_name: str,
    keep_speakers: Optional[List[str]],
    drop_silence: bool,
    order_by: Optional[str],
) -> int:
    records: List[dict] = []
    missing: List[str] = []

    for _, row in meta_df.iterrows():
        uid = str(row["uuid"]).strip()
        diag = str(row["Diagnosis"]).strip()

        sex = row.get("sex", None)
        age = row.get("age", None)
        edu = row.get("education", None)

        tsv_path = tsv_root / f"{uid}.tsv"
        if not tsv_path.exists():
            missing.append(uid)
            continue

        text = tsv_to_text(
            tsv_path,
            keep_speakers=keep_speakers,
            drop_silence=drop_silence,
            order_by=order_by,
        )

        records.append(
            {
                "ID": uid,
                "Diagnosis": diag,
                "Text_interviewer_participant": text,
                "Dataset": dataset_name,
                "Languages": "zh",
                "sex": sex,
                "age": age,
                "education": edu,
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(records)} text records to {output_jsonl}")
    if missing:
        print(f"[WARN] Missing TSV for {len(missing)} uuids (skipped). e.g. {missing[:5]}")
    return len(records)

def build_egemaps_csv(meta_df: pd.DataFrame, egemaps_df: pd.DataFrame, *, output_csv: Path) -> Tuple[int, int]:
    merged = meta_df.merge(egemaps_df, on="uuid", how="inner")

    # deterministic order
    if merged is not None and not merged.empty and "uuid" in merged.columns:
        merged = merged.sort_values(by="uuid", kind="mergesort").reset_index(drop=True)

    print(f"[INFO] Merged meta ({len(meta_df)}) with eGeMAPS ({len(egemaps_df)}) -> {len(merged)} rows.")

    feature_cols = [c for c in egemaps_df.columns if c != "uuid"]
    ordered_cols = ["uuid", "Diagnosis", "sex", "age", "education"] + feature_cols
    ordered_cols = [c for c in ordered_cols if c in merged.columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[ordered_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved eGeMAPS feature table to: {output_csv}")
    return len(merged), len(feature_cols)

# =====================================================================
# Orchestrator
# =====================================================================
def run_predictive_preprocessing(config_path: Optional[str] = None) -> Tuple[str, str]:
    cfg = load_text_config(config_path)
    pred_cfg = get_predictive_config(cfg=cfg)

    meta_csv = Path(_require(pred_cfg, "meta_csv", where="predictive"))
    egemaps_csv = Path(_require(pred_cfg, "egemaps_csv", where="predictive"))
    tsv_root = Path(_require(pred_cfg, "tsv_root", where="predictive"))
    out_text_jsonl = Path(_require(pred_cfg, "output_text_jsonl", where="predictive"))
    out_egemaps_csv = Path(_require(pred_cfg, "output_egemaps_csv", where="predictive"))

    dataset_name = str(pred_cfg.get("dataset_name", "Chinese_predictive_challenge")).strip() or "Chinese_predictive_challenge"

    # paper-strict: predictive.tsv must exist and be a dict
    tsv_cfg = _get_dict(pred_cfg, "tsv", where="predictive")

    # keep_speakers: prefer predictive.tsv.keep_speakers; fallback to predictive.keep_speakers (back-compat)
    keep_speakers = _norm_str_list(tsv_cfg.get("keep_speakers"))
    if keep_speakers is None:
        keep_speakers = _norm_str_list(pred_cfg.get("keep_speakers"))

    drop_silence = bool(tsv_cfg.get("drop_silence", True))
    order_by = tsv_cfg.get("order_by", None)
    order_by = None if order_by is None else str(order_by).strip()
    if order_by not in (None, "no", "start_time"):
        raise ValueError("predictive.tsv.order_by must be one of: 'no', 'start_time', null")

    if not meta_csv.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_csv}")
    if not egemaps_csv.exists():
        raise FileNotFoundError(f"eGeMAPS CSV not found: {egemaps_csv}")
    if not tsv_root.exists():
        raise FileNotFoundError(f"TSV root not found: {tsv_root}")

    print(f"[INFO] Loading meta from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)
    if "label" not in meta_df.columns or "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV must contain columns: uuid, label")

    meta_df, meta_stats = _normalize_and_dedup_uuid(meta_df, name="meta")
    if meta_stats["dropped_invalid"] > 0:
        print(f"[WARN] meta: dropped invalid uuid rows = {meta_stats['dropped_invalid']}")
    if meta_stats["dropped_dups"] > 0:
        print(f"[WARN] meta: dropped duplicated uuid rows = {meta_stats['dropped_dups']}")

    # deterministic order for reproducible JSONL writing
    if meta_df is not None and not meta_df.empty:
        meta_df = meta_df.sort_values(by="uuid", kind="mergesort").reset_index(drop=True)

    meta_df["Diagnosis"] = meta_df["label"].apply(lambda s: ADType.from_any(s).value)

    print(f"[INFO] Loading eGeMAPS from: {egemaps_csv}")
    egemaps_df = pd.read_csv(egemaps_csv)
    if "uuid" not in egemaps_df.columns:
        raise ValueError("eGeMAPS CSV must contain column: uuid")

    egemaps_df, eg_stats = _normalize_and_dedup_uuid(egemaps_df, name="eGeMAPS")
    if eg_stats["dropped_invalid"] > 0:
        print(f"[WARN] eGeMAPS: dropped invalid uuid rows = {eg_stats['dropped_invalid']}")
    if eg_stats["dropped_dups"] > 0:
        print(f"[WARN] eGeMAPS: dropped duplicated uuid rows = {eg_stats['dropped_dups']}")

    # deterministic order for reproducible merges/outputs
    if egemaps_df is not None and not egemaps_df.empty:
        egemaps_df = egemaps_df.sort_values(by="uuid", kind="mergesort").reset_index(drop=True)

    build_text_jsonl(
        meta_df,
        tsv_root=tsv_root,
        output_jsonl=out_text_jsonl,
        dataset_name=dataset_name,
        keep_speakers=keep_speakers,
        drop_silence=drop_silence,
        order_by=order_by,
    )
    build_egemaps_csv(meta_df, egemaps_df, output_csv=out_egemaps_csv)

    return str(out_text_jsonl), str(out_egemaps_csv)

# =====================================================================
# CLI
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess predictive dataset (paper-strict, YAML-driven).")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    return p

def cli_main() -> None:
    args = build_arg_parser().parse_args()
    run_predictive_preprocessing(config_path=args.config)

if __name__ == "__main__":
    cli_main()
