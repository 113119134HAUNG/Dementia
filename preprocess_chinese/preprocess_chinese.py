# -*- coding: utf-8 -*-
"""
preprocess_chinese.py

Config-driven preprocessing pipeline for Chinese AD text data
(text layer only, no acoustic features).

Steps
-----
1. NCMMSC ASR CSV → NCMMSC JSONL (schema aligned with other corpora)
2. Merge JSONL corpora (only include files that exist; honor target_datasets if set)
3. Normalize diagnosis labels + remove English rows
4. Structural text cleaning (Doctor:/%mor/ annotations, etc.)
5. Subset selection (dataset/labels/balancing/cap) – driven by config_text.yaml
6. Length-based outlier filtering (per Diagnosis: mean ± std)
7. Stratified train/test split → JSONL

Single source of truth
----------------------
All paths & subset rules come from `config_text.yaml`.
CLI only controls:
    --config   path to YAML
    --skip-asr reuse existing NCMMSC JSONL
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from collection import JSONLCombiner
from dataset_subset import apply_subset
from text_cleaning import clean_structured_chinese
from config_utils import get_asr_config, get_text_config

# =====================================================================
# ASR CSV → NCMMSC JSONL
# =====================================================================

def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str) -> str:
    """Convert NCMMSC ASR CSV to JSONL with unified schema."""
    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    ncmmsc_df = pd.DataFrame(
        {
            "ID": df["id"],
            "Diagnosis": df["label"],
            "Text_interviewer_participant": df["cleaned_transcript"].fillna(""),
            "Dataset": "NCMMSC2021_AD_Competition",
            "Languages": "zh",
        }
    )

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    ncmmsc_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved NCMMSC JSONL to: {jsonl_path}")
    return str(jsonl_path)

# =====================================================================
# Utilities: remove English rows, length-based filtering
# =====================================================================
def remove_english_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where Languages == 'en' (e.g., TAUKADIAL English part)."""
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]


def filter_by_length(row: pd.Series, stats: pd.DataFrame) -> bool:
    """Keep samples whose length is within mean ± std for each Diagnosis."""
    mean = stats.loc[row["Diagnosis"], "mean"]
    std = stats.loc[row["Diagnosis"], "std"]
    return mean - std <= row["length"] <= mean + std


# =====================================================================
# Merge JSONL corpora (clean + optional)
# =====================================================================
def combine_jsonls(
    *,
    ncmmsc_jsonl: str,
    predictive_jsonl: Optional[str],
    taukadial_jsonl: Optional[str],
    output_dir: str,
    merged_name: str,
    text_cfg: Dict[str, Any],
) -> str:
    """Merge multiple corpora into a single JSONL.

    Rules:
    - Only include JSONL files that exist.
    - If text_cfg has target_datasets, we only include sources that could match it
      (so your run doesn't fail when you "只用 NCMMSC").
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    target_datasets = text_cfg.get("target_datasets", None)
    target_set = set(target_datasets) if isinstance(target_datasets, list) else None

    candidates: List[Tuple[str, Optional[str]]] = [
        ("NCMMSC2021_AD_Competition", ncmmsc_jsonl),
        ("Chinese_predictive_challenge", predictive_jsonl),
        ("TAUKADIAL", taukadial_jsonl),
    ]

    input_files: List[str] = []
    for dataset_name, path_str in candidates:
        if path_str is None:
            continue

        # honor target_datasets if provided
        if target_set is not None and dataset_name not in target_set:
            continue

        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] Missing JSONL, skip: {p}")
            continue
        input_files.append(str(p))

    if not input_files:
        raise FileNotFoundError("No input JSONL files found for merging (check paths / target_datasets).")

    print("[INFO] Combining JSONL files:")
    for f in input_files:
        print(f"  - {f}")

    combiner = JSONLCombiner(input_files, str(output_dir_path), merged_name)
    combiner.combine()

    merged_path = output_dir_path / merged_name
    print(f"[INFO] Combined JSONL saved to: {merged_path}")
    return str(merged_path)

# =====================================================================
# Load + clean + subset + length filtering
# =====================================================================
def load_and_clean_chinese(merged_jsonl_path: str, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load merged JSONL and apply:
    - label normalization
    - language filtering
    - structural text cleaning
    - subset selection (config-driven)
    - length-based outlier removal
    """
    merged_jsonl_path = Path(merged_jsonl_path)
    df = pd.read_json(merged_jsonl_path, lines=True)

    if "Diagnosis" not in df.columns:
        raise ValueError("Expected column 'Diagnosis' not found in merged JSONL.")

    # Drop unknown diagnoses
    if (df["Diagnosis"] == "Unknown").any():
        unknown = df[df["Diagnosis"] == "Unknown"]
        print(
            "[WARN] Found rows with Diagnosis == 'Unknown'. Datasets:",
            set(unknown.get("Dataset", [])),
        )
        df = df[df["Diagnosis"] != "Unknown"]

    # Normalize label names (NC / CTRL → HC)
    df["Diagnosis"] = df["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})

    # Keep only non-English rows
    df = remove_english_rows(df)

    # Structural text cleaning
    if "Text_interviewer_participant" not in df.columns:
        raise ValueError("Expected column 'Text_interviewer_participant' not found.")
    df["Text_interviewer_participant"] = df["Text_interviewer_participant"].apply(clean_structured_chinese)

    # Subset selection (dataset/labels/balance/cap) driven by YAML
    df = apply_subset(df, text_cfg)
    print(f"[INFO] After subset: {len(df)} samples remaining.")
    if not df.empty:
        print("[INFO] Label distribution after subset:\n", df["Diagnosis"].value_counts())

    # Length stats and filtering (computed AFTER subset)
    if df.empty:
        return df

    df["length"] = df["Text_interviewer_participant"].apply(len)
    length_stats = df.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
    print("[INFO] Length stats by Diagnosis:\n", length_stats)

    df = df[df.apply(filter_by_length, axis=1, stats=length_stats)]
    print(f"[INFO] After length filtering: {len(df)} samples remaining.")
    if not df.empty:
        print("[INFO] Label distribution after filtering:\n", df["Diagnosis"].value_counts())

    return df

# =====================================================================
# Train/test split + save
# =====================================================================
def split_and_save(
    df: pd.DataFrame,
    output_dir: str,
    train_name: str,
    test_name: str,
) -> Tuple[str, str]:
    """Stratified train/test split and save as JSONL."""
    if df.empty:
        raise ValueError("No samples to split (df is empty).")

    vc = df["Diagnosis"].value_counts()
    if vc.size < 2:
        raise ValueError(f"Need at least 2 classes for stratified split, got: {list(vc.index)}")
    if (vc < 2).any():
        raise ValueError(f"Each class must have >=2 samples for split. Counts:\n{vc}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Diagnosis"],
        random_state=42,
    )

    train_out = output_dir_path / train_name
    test_out = output_dir_path / test_name

    train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)

    print(f"[INFO] Saved train split to: {train_out}")
    print(f"[INFO] Saved test  split to: {test_out}")
    return str(train_out), str(test_out)

# =====================================================================
# Orchestrator
# =====================================================================
def run_chinese_preprocessing(
    config_path: Optional[str] = None,
    skip_asr: bool = False,
) -> Tuple[str, str]:
    """End-to-end preprocessing for Chinese AD datasets."""
    asr_cfg = get_asr_config(path=config_path)
    text_cfg = get_text_config(path=config_path)

    # Step 1: ASR CSV → NCMMSC JSONL
    ncmmsc_jsonl_path = text_cfg["ncmmsc_jsonl"]
    if not skip_asr:
        asr_csv_path = asr_cfg["output_csv"]
        csv_to_ncmmsc_jsonl(asr_csv_path, ncmmsc_jsonl_path)
    else:
        print(f"[INFO] Skipping ASR → JSONL. Using existing: {ncmmsc_jsonl_path}")

    # Step 2: merge corpora (clean: optional files)
    merged_path = combine_jsonls(
        ncmmsc_jsonl=ncmmsc_jsonl_path,
        predictive_jsonl=text_cfg.get("predictive_jsonl"),
        taukadial_jsonl=text_cfg.get("taukadial_jsonl"),
        output_dir=text_cfg["output_dir"],
        merged_name=text_cfg["combined_name"],
        text_cfg=text_cfg,
    )

    # Step 3–7: clean + subset + length filtering + split & save
    df_clean = load_and_clean_chinese(merged_path, text_cfg)
    return split_and_save(
        df_clean,
        output_dir=text_cfg["output_dir"],
        train_name=text_cfg["train_jsonl"],
        test_name=text_cfg["test_jsonl"],
    )

# =====================================================================
# CLI entry point
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Chinese preprocessing."""
    parser = argparse.ArgumentParser(description=("Preprocess Chinese AD datasets (text only) – config-driven."))
    parser.add_argument("--config",type=str,default=None,help="Path to config_text.yaml（預設：專案根目錄的 config_text.yaml）",)
    parser.add_argument("--skip-asr",action="store_true",help="Skip ASR CSV→JSONL step and reuse existing NCMMSC JSONL.",)
    return parser

def cli_main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_chinese_preprocessing(
        config_path=args.config,
        skip_asr=args.skip_asr,
    )

if __name__ == "__main__":
    cli_main()
