# -*- coding: utf-8 -*-
"""
preprocess_predictive.py

Config-driven preprocessing for the Chinese predictive challenge dataset.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from enums import ADType
from config_utils import get_predictive_config

PREDICTIVE_DATASET_NAME = "Chinese_predictive_challenge"
KEEP_SPEAKERS: Optional[List[str]] = None

# =====================================================================
# Small utils (keep clean, no extra deps)
# =====================================================================
def _normalize_and_dedup_uuid(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize uuid as stripped string and drop duplicated uuids (keep first)."""
    if df is None or df.empty:
        return df
    if "uuid" not in df.columns:
        return df

    out = df.copy()
    out["uuid"] = out["uuid"].astype(str).str.strip()
    out = out.dropna(subset=["uuid"])

    dup = int(out["uuid"].duplicated().sum())
    if dup > 0:
        print(f"[WARN] {name}: found {dup} duplicated uuid rows → keep first occurrence.")
        out = out.drop_duplicates(subset=["uuid"], keep="first")

    return out.reset_index(drop=True)

# =====================================================================
# TSV → long-form text
# =====================================================================
def tsv_to_text(tsv_path: Path) -> str:
    """Convert a single *.tsv file into one long text string."""
    df = pd.read_csv(tsv_path, sep="\t", keep_default_na=False)

    if "value" not in df.columns:
        raise ValueError(f"TSV file {tsv_path} has no 'value' column.")

    if KEEP_SPEAKERS is not None:
        if "speaker" not in df.columns:
            raise ValueError(f"TSV file {tsv_path} has no 'speaker' column.")
        df = df[df["speaker"].astype(str).isin(KEEP_SPEAKERS)]

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
def build_text_jsonl(meta_df: pd.DataFrame, tsv_root: Path, output_jsonl: Path) -> int:
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

        text = tsv_to_text(tsv_path)

        records.append(
            {
                "ID": uid,
                "Diagnosis": diag,
                "Text_interviewer_participant": text,
                "Dataset": PREDICTIVE_DATASET_NAME,
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
        print(f"[WARN] Missing TSV for {len(missing_tsv)} uuids (skipped).")
        print("       e.g.", missing_tsv[:5])
    return len(records)

# =====================================================================
# eGeMAPS CSV: meta + eGeMAPS → feature CSV
# =====================================================================
def build_egemaps_csv(meta_df: pd.DataFrame, egemaps_df: pd.DataFrame, output_csv: Path) -> Tuple[int, int]:
    if "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'uuid' column.")
    if "uuid" not in egemaps_df.columns:
        raise ValueError("eGeMAPS CSV is expected to contain a 'uuid' column.")

    merged = meta_df.merge(egemaps_df, on="uuid", how="inner")
    print(
        f"[INFO] Merged meta ({len(meta_df)}) with eGeMAPS ({len(egemaps_df)}) "
        f"→ {len(merged)} rows."
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

    meta_csv = Path(pred_cfg["meta_csv"])
    egemaps_csv = Path(pred_cfg["egemaps_csv"])
    tsv_root = Path(pred_cfg["tsv_root"])
    out_text_jsonl = Path(pred_cfg["output_text_jsonl"])
    out_egemaps_csv = Path(pred_cfg["output_egemaps_csv"])

    print(f"[INFO] Loading meta from: {meta_csv}")
    meta_df = pd.read_csv(meta_csv)

    if "label" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'label' column.")

    # Normalize & de-duplicate uuid ONCE (paper-like single point)
    meta_df = _normalize_and_dedup_uuid(meta_df, name="meta")

    # Normalize labels to canonical 3-way: AD / HC / MCI
    meta_df["Diagnosis"] = meta_df["label"].apply(lambda s: ADType.from_any(s).value)

    print(f"[INFO] Loading eGeMAPS from: {egemaps_csv}")
    egemaps_df = pd.read_csv(egemaps_csv)

    # Normalize & de-duplicate uuid ONCE
    egemaps_df = _normalize_and_dedup_uuid(egemaps_df, name="eGeMAPS")

    build_text_jsonl(meta_df, tsv_root, out_text_jsonl)
    build_egemaps_csv(meta_df, egemaps_df, out_egemaps_csv)

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
