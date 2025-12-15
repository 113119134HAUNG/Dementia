# -*- coding: utf-8 -*-
"""
preprocess_chinese.py

Config-driven preprocessing pipeline for Chinese AD text data (text only).

Steps
-----
1) NCMMSC ASR CSV → NCMMSC JSONL (unified schema)
2) Merge JSONL corpora (only include files that exist)
3) Normalize diagnosis labels + remove English rows
4) Text cleaning (single point)
5) Subset selection (dataset/labels/balancing/cap) – YAML-driven
6) Length-based outlier filtering (per Diagnosis: mean ± std_k * std) – YAML-driven
7) Stratified train/test split → JSONL – YAML-driven

Single source of truth
----------------------
All paths & numeric knobs come from `config_text.yaml`:
- text.length_filter.std_k
- text.split.test_size
- text.split.random_state
- text.split.min_per_class

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
from text_cleaning import clean_asr_chinese, clean_structured_chinese
from config_utils import get_asr_config, get_text_config

NCMMSC_DATASET_NAME = "NCMMSC2021_AD_Competition"
PREDICTIVE_DATASET_NAME = "Chinese_predictive_challenge"
TAUKADIAL_DATASET_NAME = "TAUKADIAL"

# =====================================================================
# Config helpers (strict)
# =====================================================================

def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Config missing required key: {key}")
    return cfg[key]


def _get_length_filter_cfg(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    lf = _require(text_cfg, "length_filter")
    if not isinstance(lf, dict):
        raise ValueError("text.length_filter must be a dict.")
    _require(lf, "std_k")
    return lf


def _get_split_cfg(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sp = _require(text_cfg, "split")
    if not isinstance(sp, dict):
        raise ValueError("text.split must be a dict.")
    _require(sp, "test_size")
    _require(sp, "random_state")
    _require(sp, "min_per_class")
    return sp

# =====================================================================
# Step 1  ASR CSV → NCMMSC JSONL
# =====================================================================
def _select_asr_text_column(df: pd.DataFrame) -> str:
    # Strict: prefer raw transcript (so cleaning happens once later)
    if "transcript" in df.columns:
        return "transcript"
    if "cleaned_transcript" in df.columns:
        print("[WARN] ASR CSV missing 'transcript'; fallback to 'cleaned_transcript'.")
        return "cleaned_transcript"
    raise ValueError("ASR CSV must contain 'transcript' (preferred) or 'cleaned_transcript'.")


def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str) -> str:
    """Convert NCMMSC ASR CSV to JSONL with unified schema."""
    csv_path_p = Path(csv_path)
    jsonl_path_p = Path(jsonl_path)

    df = pd.read_csv(csv_path_p, encoding="utf-8-sig")

    for col in ("id", "label"):
        if col not in df.columns:
            raise ValueError(f"ASR CSV missing required column: {col}")

    text_col = _select_asr_text_column(df)

    out_df = pd.DataFrame(
        {
            "ID": df["id"].astype(str),
            "Diagnosis": df["label"].astype(str),
            "Text_interviewer_participant": df[text_col].fillna("").astype(str),
            "Dataset": NCMMSC_DATASET_NAME,
            "Languages": "zh",
        }
    )

    jsonl_path_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(jsonl_path_p, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved NCMMSC JSONL to: {jsonl_path_p}")
    return str(jsonl_path_p)

# =====================================================================
# Step 2  Merge JSONLs (only existing)
# =====================================================================
def _collect_existing_jsonls(
    candidates: List[Tuple[str, Optional[str]]]
) -> List[str]:
    files: List[str] = []
    for _, path_str in candidates:
        if not path_str:
            continue
        p = Path(path_str)
        if not p.exists():
            print(f"[WARN] Missing JSONL, skip: {p}")
            continue
        files.append(str(p))
    return files

def combine_jsonls(
    *,
    ncmmsc_jsonl: str,
    predictive_jsonl: Optional[str],
    taukadial_jsonl: Optional[str],
    output_dir: str,
    merged_name: str,
) -> str:
    """Merge corpora into a single JSONL (include only files that exist)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: List[Tuple[str, Optional[str]]] = [
        (NCMMSC_DATASET_NAME, ncmmsc_jsonl),
        (PREDICTIVE_DATASET_NAME, predictive_jsonl),
        (TAUKADIAL_DATASET_NAME, taukadial_jsonl),
    ]

    input_files = _collect_existing_jsonls(candidates)
    if not input_files:
        raise FileNotFoundError("No input JSONL files found for merging (check config paths).")

    print("[INFO] Combining JSONL files:")
    for f in input_files:
        print(f"  - {f}")

    combiner = JSONLCombiner(input_files, str(out_dir), merged_name)
    combiner.combine()

    merged_path = out_dir / merged_name
    print(f"[INFO] Combined JSONL saved to: {merged_path}")
    return str(merged_path)

# =====================================================================
# Step 3–6 Load + normalize + clean + subset + length filter#  
# =====================================================================
def remove_english_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]


def normalize_diagnosis_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "Diagnosis" not in df.columns:
        raise ValueError("Expected column 'Diagnosis' not found.")
    out = df.copy()
    # Drop Unknown
    if (out["Diagnosis"] == "Unknown").any():
        unk = out[out["Diagnosis"] == "Unknown"]
        print("[WARN] Found Diagnosis == 'Unknown'. Datasets:", set(unk.get("Dataset", [])))
        out = out[out["Diagnosis"] != "Unknown"]
    # NC/CTRL → HC
    out["Diagnosis"] = out["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})
    return out


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Text_interviewer_participant" not in df.columns:
        raise ValueError("Expected column 'Text_interviewer_participant' not found.")
    out = df.copy()
    # single cleaning point: ASR-level light clean → structured clean
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].apply(clean_asr_chinese)
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].apply(clean_structured_chinese)
    return out


def length_filter(
    df: pd.DataFrame,
    *,
    std_k: float,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["length"] = out["Text_interviewer_participant"].apply(len)

    stats = out.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
    print("[INFO] Length stats by Diagnosis:\n", stats)

    # std could be NaN if class has 1 sample
    stats["std"] = stats["std"].fillna(0.0)

    merged = out.merge(
        stats[["mean", "std"]],
        left_on="Diagnosis",
        right_index=True,
        how="left",
        suffixes=("", "_stat"),
    )

    lower = merged["mean"] - (std_k * merged["std"])
    upper = merged["mean"] + (std_k * merged["std"])
    keep = (merged["length"] >= lower) & (merged["length"] <= upper)

    filtered = out.loc[keep.values].reset_index(drop=True)
    print(f"[INFO] After length filtering: {len(filtered)} samples remaining.")
    if not filtered.empty:
        print("[INFO] Label distribution after filtering:\n", filtered["Diagnosis"].value_counts())
    return filtered


def load_and_process_chinese(
    merged_jsonl_path: str,
    text_cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Load merged JSONL and run Steps 3–6."""
    df = pd.read_json(Path(merged_jsonl_path), lines=True)

    df = normalize_diagnosis_labels(df)
    df = remove_english_rows(df)
    df = clean_text_column(df)

    # YAML-driven subset (dataset/labels/balance/cap)
    df = apply_subset(df, text_cfg)
    print(f"[INFO] After subset: {len(df)} samples remaining.")
    if not df.empty:
        print("[INFO] Label distribution after subset:\n", df["Diagnosis"].value_counts())

    if df.empty:
        return df.reset_index(drop=True)

    lf_cfg = _get_length_filter_cfg(text_cfg)
    std_k = float(lf_cfg["std_k"])
    return length_filter(df, std_k=std_k)


# =====================================================================
# Step 7: Split + save (YAML-driven)
# =====================================================================

def _validate_split_feasibility(df: pd.DataFrame, min_per_class: int) -> None:
    if df.empty:
        raise ValueError("No samples to split (df is empty).")
    if "Diagnosis" not in df.columns:
        raise ValueError("Expected column 'Diagnosis' not found.")
    vc = df["Diagnosis"].value_counts()
    if vc.size < 2:
        raise ValueError(f"Need at least 2 classes for stratified split, got: {list(vc.index)}")
    if (vc < int(min_per_class)).any():
        raise ValueError(f"Each class must have >= {min_per_class} samples. Counts:\n{vc}")

def split_and_save(
    df: pd.DataFrame,
    *,
    output_dir: str,
    train_name: str,
    test_name: str,
    test_size: float,
    random_state: int,
    min_per_class: int,
) -> Tuple[str, str]:
    _validate_split_feasibility(df, min_per_class=min_per_class)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=float(test_size),
        stratify=df["Diagnosis"],
        random_state=int(random_state),
    )

    train_out = out_dir / train_name
    test_out = out_dir / test_name

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
    asr_cfg = get_asr_config(path=config_path)
    text_cfg = get_text_config(path=config_path)

    # required text keys (strict)
    ncmmsc_jsonl = _require(text_cfg, "ncmmsc_jsonl")
    output_dir = _require(text_cfg, "output_dir")
    merged_name = _require(text_cfg, "combined_name")
    train_name = _require(text_cfg, "train_jsonl")
    test_name = _require(text_cfg, "test_jsonl")

    # Step 1
    if not skip_asr:
        asr_csv = _require(asr_cfg, "output_csv")
        csv_to_ncmmsc_jsonl(asr_csv, ncmmsc_jsonl)
    else:
        print(f"[INFO] Skipping ASR → JSONL. Using existing: {ncmmsc_jsonl}")

    # Step 2
    merged_path = combine_jsonls(
        ncmmsc_jsonl=ncmmsc_jsonl,
        predictive_jsonl=text_cfg.get("predictive_jsonl"),
        taukadial_jsonl=text_cfg.get("taukadial_jsonl"),
        output_dir=output_dir,
        merged_name=merged_name,
    )

    # Step 3–6
    df_clean = load_and_process_chinese(merged_path, text_cfg)

    # Step 7 (YAML-driven)
    sp_cfg = _get_split_cfg(text_cfg)
    return split_and_save(
        df_clean,
        output_dir=output_dir,
        train_name=train_name,
        test_name=test_name,
        test_size=float(sp_cfg["test_size"]),
        random_state=int(sp_cfg["random_state"]),
        min_per_class=int(sp_cfg["min_per_class"]),
    )

# =====================================================================
# CLI
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Chinese AD datasets (text only) – config-driven.")
    parser.add_argument("--config",type=str,default=None,help="Path to config_text.yaml（預設：專案根目錄的 config_text.yaml）",)
    parser.add_argument("--skip-asr",action="store_true",help="Skip ASR CSV→JSONL step and reuse existing NCMMSC JSONL.",)
    return parser

def cli_main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_chinese_preprocessing(config_path=args.config, skip_asr=args.skip_asr)

if __name__ == "__main__":
    cli_main()
