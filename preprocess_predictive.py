# -*- coding: utf-8 -*-
"""
preprocess_predictive.py

Config-driven preprocessing for the Chinese predictive challenge dataset.

Input resources
---------------
1. Meta CSV (e.g., 2_final_list_train.csv)
       - columns: uuid, label, sex, age, education, ...
2. eGeMAPS CSV (e.g., egemaps_final.csv)
       - one row per uuid, many acoustic features
3. Per-utterance TSV transcripts
       - files named as <uuid>.tsv
       - schema: no, start_time, end_time, speaker, value

Outputs
-------
1. Text JSONL for downstream NLP:
       predictive.output_text_jsonl
       (to be used later as `text.predictive_jsonl` in preprocess_chinese.py)
2. eGeMAPS feature CSV for acoustic baselines:
       predictive.output_egemaps_csv

All paths are configured in config_text.yaml under the `predictive` section.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from enums import ADType
from config_utils import get_predictive_config

# Dataset name used in the JSONL "Dataset" field
PREDICTIVE_DATASET_NAME = "Chinese_predictive_challenge"

# If you want to keep only certain speakers (e.g., patient channel),
# set this to a list like ["<B>"] or ["<A>"].
# None = keep all non-"sil" rows regardless of speaker.
KEEP_SPEAKERS: Optional[List[str]] = None

# =====================================================================
#    TSV → long-form text
# =====================================================================

def tsv_to_text(tsv_path: Path) -> str:
    """Convert a single *.tsv file into one long text string.

    Expected TSV schema
    -------------------
    Columns:
        - no
        - start_time
        - end_time
        - speaker
        - value
    """
    df = pd.read_csv(tsv_path, sep="\t")

    if "value" not in df.columns:
        raise ValueError(f"TSV file {tsv_path} has no 'value' column.")

    # Drop pure silence rows
    df = df[df["value"].astype(str) != "sil"]

    # Optionally keep only specific speakers
    if KEEP_SPEAKERS is not None:
        if "speaker" not in df.columns:
            raise ValueError(f"TSV file {tsv_path} has no 'speaker' column.")
        df = df[df["speaker"].isin(KEEP_SPEAKERS)]

    # Simple concatenation of 'value' (keep &嗯 / annotations as-is;
    # downstream text_cleaning will decide what to remove)
    text = " ".join(df["value"].astype(str))
    return text

# =====================================================================
#    Text JSONL: meta + TSV → JSONL
# =====================================================================

def build_text_jsonl(
    meta_df: pd.DataFrame,
    tsv_root: Path,
    output_jsonl: Path,
) -> int:
    """Build predictive text JSONL from meta CSV and TSV files.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Meta table including `uuid` and `Diagnosis` (AD / HC / MCI),
        plus optional demographics (sex, age, education).
    tsv_root : Path
        Directory containing <uuid>.tsv transcripts.
    output_jsonl : Path
        Output JSONL path.

    Returns
    -------
    int
        Number of successfully processed samples.
    """
    records: List[dict] = []
    missing_tsv: List[str] = []

    if "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'uuid' column.")
    if "Diagnosis" not in meta_df.columns:
        raise ValueError("Meta DataFrame is expected to contain 'Diagnosis'.")

    for _, row in meta_df.iterrows():
        uid = row["uuid"]
        diag = row["Diagnosis"]
        sex = row.get("sex", None)
        age = row.get("age", None)
        edu = row.get("education", None)

        tsv_path = tsv_root / f"{uid}.tsv"
        if not tsv_path.exists():
            missing_tsv.append(uid)
            continue

        text = tsv_to_text(tsv_path)

        rec = {
            "ID": uid,
            "Diagnosis": diag,
            "Text_interviewer_participant": text,
            "Dataset": PREDICTIVE_DATASET_NAME,
            "Languages": "zh",
            "sex": sex,
            "age": age,
            "education": edu,
        }
        records.append(rec)

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
#   eGeMAPS CSV: meta + eGeMAPS → feature CSV
# =====================================================================

def build_egemaps_csv(
    meta_df: pd.DataFrame,
    egemaps_df: pd.DataFrame,
    output_csv: Path,
) -> Tuple[int, int]:
    """Merge meta and eGeMAPS tables on `uuid` and save feature CSV.

    Output columns (in order)
    -------------------------
        - uuid
        - Diagnosis
        - sex, age, education
        - all eGeMAPS features from egemaps_df (excluding its own 'uuid')
    """
    if "uuid" not in meta_df.columns:
        raise ValueError("Meta CSV is expected to contain a 'uuid' column.")
    if "uuid" not in egemaps_df.columns:
        raise ValueError("eGeMAPS CSV is expected to contain a 'uuid' column.")

    merged = meta_df.merge(egemaps_df, on="uuid", how="inner")
    print(
        f"[INFO] Merged meta ({len(meta_df)}) with eGeMAPS ({len(egemaps_df)}) "
        f"→ {len(merged)} rows."
    )

    # Basic info first, acoustic features afterwards
    feature_cols = [c for c in egemaps_df.columns if c != "uuid"]
    ordered_cols = ["uuid", "Diagnosis", "sex", "age", "education"] + feature_cols
    ordered_cols = [c for c in ordered_cols if c in merged.columns]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[ordered_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved eGeMAPS feature table to: {output_csv}")
    return len(merged), len(feature_cols)

# =====================================================================
#    Orchestrator
# =====================================================================

def run_predictive_preprocessing(
    config_path: Optional[str] = None,
) -> Tuple[str, str]:
    """End-to-end preprocessing for the Chinese predictive challenge dataset.

    Parameters
    ----------
    config_path : str or None
        Path to config_text.yaml. If None, the default path is used.

    Returns
    -------
    (str, str)
        Tuple of (output_text_jsonl_path, output_egemaps_csv_path).
    """
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

    # Normalize labels to AD / HC / MCI using ADType
    meta_df["Diagnosis"] = meta_df["label"].apply(
        lambda s: ADType.from_any(s).value
    )

    print(f"[INFO] Loading eGeMAPS from: {egemaps_csv}")
    egemaps_df = pd.read_csv(egemaps_csv)

    # 1) Build text JSONL for NLP experiments
    build_text_jsonl(meta_df, tsv_root, out_text_jsonl)

    # 2) Build eGeMAPS feature CSV for acoustic experiments
    build_egemaps_csv(meta_df, egemaps_df, out_egemaps_csv)

    return str(out_text_jsonl), str(out_egemaps_csv)

# =====================================================================
#    CLI
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for predictive preprocessing."""
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
