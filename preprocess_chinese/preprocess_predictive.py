# -*- coding: utf-8 -*-
"""
preprocess_predictive.py

Config-driven preprocessing for the Chinese predictive challenge dataset.

Outputs
-------
1) Text JSONL (unified schema for downstream text pipeline)
2) eGeMAPS feature CSV (meta + acoustic features)

Single source of truth
----------------------
All paths & optional knobs come from config_text.yaml under `predictive`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from enums import ADType
from config_utils import get_predictive_config

PREDICTIVE_DATASET_NAME = "Chinese_predictive_challenge"

# =====================================================================
# Strict config helpers
# =====================================================================
def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Config missing required key: predictive.{key}")
    return cfg[key]


def _get_keep_speakers(pred_cfg: Dict[str, Any]) -> Optional[List[str]]:
    """Optional config: predictive.keep_speakers (null or list[str])."""
    if "keep_speakers" not in pred_cfg or pred_cfg.get("keep_speakers") is None:
        return None

    v = pred_cfg.get("keep_speakers")
    if isinstance(v, (list, tuple)):
        out = [str(x).strip() for x in v if str(x).strip()]
        return out or None
    s = str(v).strip()
    return [s] if s else None

# =====================================================================
# Small utils (uuid normalize + dedup) - single point
# =====================================================================
_INVALID_UUID = {"", "nan", "none", "null"}


def _normalize_and_dedup_uuid(df: pd.DataFrame, *, name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Normalize uuid and drop duplicated uuids (keep first).

    - uuid -> stripped string
    - drop invalid uuid: "", "nan", "none", "null" (case-insensitive)
    - drop duplicate uuid rows (keep first)

    Returns
    -------
    (clean_df, stats)
      stats: {"dropped_invalid": int, "dropped_dups": int}
    """
    if df is None or df.empty:
        return df, {"dropped_invalid": 0, "dropped_dups": 0}
    if "uuid" not in df.columns:
        return df, {"dropped_invalid": 0, "dropped_dups": 0}

    out = df.copy()

    # Normalize uuid
    uu = out["uuid"].astype(str).str.strip()
    uu_low = uu.str.lower()
    valid = ~uu_low.isin(_INVALID_UUID)
    dropped_invalid = int((~valid).sum())
    out = out.loc[valid].copy()
    out["uuid"] = uu.loc[valid].astype(str).str.strip()

    # Dedup
    dup_mask = out["uuid"].duplicated(keep="first")
    dropped_dups = int(dup_mask.sum())
    if dropped_dups > 0:
        out = out.loc[~dup_mask].copy()

    return out.reset_index(drop=True), {"dropped_invalid": dropped_invalid, "dropped_dups": dropped_dups}

# =====================================================================
# TSV → long-form text
# =====================================================================
def tsv_to_text(tsv_path: Path, *, keep_speakers: Optional[List[str]] = None) -> str:
    """Convert a single *.tsv file into one long text string."""
    df = pd.read_csv(tsv_path, sep="\t", keep_default_na=False)

    if "value" not in df.columns:
        raise ValueError(f"TSV file {tsv_path} has no 'value' column.")

    # Optional speaker filter
    if keep_speakers is not None:
        if "speaker" not in df.columns:
            raise ValueError(f"TSV file {tsv_path} has no 'speaker' column.")
        df = df[df["speaker"].astype(str).str.strip().isin([str(x).strip() for x in keep_speakers])]

    # Prefer stable ordering if possible
    if "no" in df.columns:
        df["_no"] = pd.to_numeric(df["no"], errors="coerce")
        df = df.sort_values(by="_no", kind="mergesort").drop(columns=["_no"])
    elif "start_time" in df.columns:
        df["_st"] = pd.to_numeric(df["start_time"], errors="coerce")
        df = df.sort_values(by="_st", kind="mergesort").drop(columns=["_st"])

    vals = df["value"].astype(str).str.strip()
    vals = vals[vals != ""]
    vals = vals[vals.str.lower() != "sil"]

    return " ".join(vals.tolist())

# =====================================================================
# Text JSONL: meta + TSV → JSONL
# =====================================================================
def build_text_jsonl(
    meta_df: pd.DataFrame,
    *,
    tsv_root: Path,
    output_jsonl: Path,
    keep_speakers: Optional[List[str]],
    dataset_name: str = PREDICTIVE_DATASET_NAME,
) -> int:
    records: List[dict] = []
    missing_tsv: List[str] = []

    if "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'uuid' column.")
    if "Diagnosis" not in meta_df.columns:
        raise ValueError("Meta DataFrame is expected to contain 'Diagnosis'.")

    for _, row in meta_df.iterrows():
        uid = str(row["uuid"]).strip()
        diag = str(row["Diagnosis"]).strip()

        sex = row.get("sex", None)
        age = row.get("age", None)
        edu = row.get("education", None)

        tsv_path = tsv_root / f"{uid}.tsv"
        if not tsv_path.exists():
            missing_tsv.append(uid)
            continue

        text = tsv_to_text(tsv_path, keep_speakers=keep_speakers)

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
    if missing_tsv:
        print(f"[WARN] Missing TSV for {len(missing_tsv)} uuids (skipped). e.g. {missing_tsv[:5]}")
    return len(records)

# =====================================================================
# eGeMAPS CSV: meta + eGeMAPS → feature CSV
# =====================================================================
def build_egemaps_csv(
    meta_df: pd.DataFrame,
    egemaps_df: pd.DataFrame,
    *,
    output_csv: Path,
) -> Tuple[int, int]:
    if "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'uuid' column.")
    if "uuid" not in egemaps_df.columns:
        raise ValueError("eGeMAPS CSV is expected to contain a 'uuid' column.")

    merged = meta_df.merge(egemaps_df, on="uuid", how="inner")
    print(
        f"[INFO] Merged meta ({len(meta_df)}) with eGeMAPS ({len(egemaps_df)}) → {len(merged)} rows."
    )

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
    pred_cfg = get_predictive_config(path=config_path)

    meta_csv = Path(_require(pred_cfg, "meta_csv"))
    egemaps_csv = Path(_require(pred_cfg, "egemaps_csv"))
    tsv_root = Path(_require(pred_cfg, "tsv_root"))
    out_text_jsonl = Path(_require(pred_cfg, "output_text_jsonl"))
    out_egemaps_csv = Path(_require(pred_cfg, "output_egemaps_csv"))

    keep_speakers = _get_keep_speakers(pred_cfg)

    if not meta_csv.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_csv}")
    if not egemaps_csv.exists():
        raise FileNotFoundError(f"eGeMAPS CSV not found: {egemaps_csv}")
    if not tsv_root.exists():
        raise FileNotFoundError(f"TSV root not found: {tsv_root}")

    print(f"[INFO] Loading meta from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)

    if "label" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'label' column.")
    if "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'uuid' column.")

    # uuid normalize + dedup (single point)
    meta_df, meta_stats = _normalize_and_dedup_uuid(meta_df, name="meta")
    if meta_stats["dropped_invalid"] > 0:
        print(f"[WARN] meta: dropped invalid uuid rows = {meta_stats['dropped_invalid']}")
    if meta_stats["dropped_dups"] > 0:
        print(f"[WARN] meta: dropped duplicated uuid rows = {meta_stats['dropped_dups']}")

    # Normalize labels to canonical 3-way: AD / HC / MCI
    meta_df["Diagnosis"] = meta_df["label"].apply(lambda s: ADType.from_any(s).value)

    print(f"[INFO] Loading eGeMAPS from: {egemaps_csv}")
    egemaps_df = pd.read_csv(egemaps_csv)

    if "uuid" not in egemaps_df.columns:
        raise ValueError("eGeMAPS CSV is expected to contain a 'uuid' column.")

    egemaps_df, eg_stats = _normalize_and_dedup_uuid(egemaps_df, name="eGeMAPS")
    if eg_stats["dropped_invalid"] > 0:
        print(f"[WARN] eGeMAPS: dropped invalid uuid rows = {eg_stats['dropped_invalid']}")
    if eg_stats["dropped_dups"] > 0:
        print(f"[WARN] eGeMAPS: dropped duplicated uuid rows = {eg_stats['dropped_dups']}")

    build_text_jsonl(
        meta_df,
        tsv_root=tsv_root,
        output_jsonl=out_text_jsonl,
        keep_speakers=keep_speakers,
        dataset_name=PREDICTIVE_DATASET_NAME,
    )
    build_egemaps_csv(meta_df, egemaps_df, output_csv=out_egemaps_csv)

    return str(out_text_jsonl), str(out_egemaps_csv)

# =====================================================================
# CLI
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess the Chinese predictive challenge dataset "
            "(meta + TSV transcripts + eGeMAPS) – config-driven."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config_text.yaml（預設：專案根目錄的 config_text.yaml）",
    )
    return parser

def cli_main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_predictive_preprocessing(config_path=args.config)

if __name__ == "__main__":
    cli_main()
