# -*- coding: utf-8 -*-
"""
preprocess/preprocess_chinese.py (paper-strict)

Step 0 (optional):
- Rebuild NCMMSC ASR JSONL from ASR CSV (id,label,transcript,audio_path,duration)
  controlled by CLI flag: --force-asr-jsonl

Step 1-2:
- Load YAML once.
- Load corpora JSONL listed in text.corpora (only those in text.target_datasets).
- Merge into a single combined JSONL (deterministic order).
- Call chinese_steps.load_and_process_chinese (Step 3-12).
- Save cleaned.jsonl.

This file should NOT do any cleaning itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from preprocess.chinese_steps import load_and_process_chinese, save_cleaned_jsonl
from tools.config_utils import load_text_config

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

def _get_list(cfg: Dict[str, Any], key: str, *, where: str = "") -> List[Any]:
    v = _require(cfg, key, where=where)
    if not isinstance(v, list):
        prefix = f"{where}." if where else ""
        raise ValueError(f"{prefix}{key} must be a list.")
    return v

def _normalize_schema(df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    out = df.copy()

    # Dataset
    if "Dataset" not in out.columns:
        out["Dataset"] = dataset_name
    else:
        out["Dataset"] = out["Dataset"].astype(str).fillna("").replace("", dataset_name)

    # ID
    if "ID" not in out.columns and "id" in out.columns:
        out = out.rename(columns={"id": "ID"})
    if "ID" not in out.columns:
        raise ValueError(f"[{dataset_name}] Missing ID column (expected 'ID' or 'id').")
    out["ID"] = out["ID"].astype(str).str.strip()

    # Diagnosis
    if "Diagnosis" not in out.columns and "label" in out.columns:
        out = out.rename(columns={"label": "Diagnosis"})
    if "Diagnosis" not in out.columns:
        raise ValueError(f"[{dataset_name}] Missing Diagnosis column (expected 'Diagnosis' or 'label').")
    out["Diagnosis"] = out["Diagnosis"].astype(str).str.strip()

    # Text
    if "Text_interviewer_participant" not in out.columns and "transcript" in out.columns:
        out = out.rename(columns={"transcript": "Text_interviewer_participant"})
    if "Text_interviewer_participant" not in out.columns:
        raise ValueError(
            f"[{dataset_name}] Missing text column (expected 'Text_interviewer_participant' or 'transcript')."
        )
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].fillna("").astype(str)

    # Languages (optional)
    if "Languages" not in out.columns:
        out["Languages"] = "zh"
    else:
        out["Languages"] = out["Languages"].fillna("").astype(str)
        out.loc[out["Languages"].str.strip() == "", "Languages"] = "zh"

    # Deterministic row order
    out = out.sort_values(by=["Dataset", "ID"], kind="mergesort").reset_index(drop=True)
    return out

def rebuild_ncmmsc_jsonl_from_asr_csv(
    *,
    asr_cfg: Dict[str, Any],
    text_cfg: Dict[str, Any],
    force: bool,
    dataset_name: str = "NCMMSC2021_AD_Competition",
) -> Optional[str]:
    """
    Build / overwrite text.ncmmsc_jsonl from asr.output_csv.

    ASR CSV expected columns:
      id,label,transcript,audio_path,duration
    """
    out_jsonl = text_cfg.get("ncmmsc_jsonl", None)
    if not out_jsonl:
        # if user didn't configure it, just skip
        return None

    out_path = Path(str(out_jsonl)).expanduser()
    csv_path = Path(str(_require(asr_cfg, "output_csv", where="asr"))).expanduser()

    if out_path.exists() and not force:
        print(f"[INFO] NCMMSC ASR JSONL exists -> skip rebuild (force=False): {out_path}")
        return str(out_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"[asr.output_csv] not found: {csv_path}")

    print(f"[INFO] Rebuilding NCMMSC ASR JSONL from CSV (force={force})")
    df = pd.read_csv(csv_path)

    # Normalize to pipeline schema + keep audio_path/duration if present
    df = _normalize_schema(df, dataset_name=dataset_name)

    # keep optional columns if present (no harm)
    for col in ["audio_path", "duration"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Built NCMMSC ASR JSONL: {out_path} (n={len(df)})")
    return str(out_path)

def merge_corpora_to_jsonl(*, text_cfg: Dict[str, Any]) -> str:
    output_dir = Path(str(_require(text_cfg, "output_dir", where="text"))).expanduser()
    combined_name = str(_require(text_cfg, "combined_name", where="text")).strip()
    combined_path = output_dir / combined_name

    corpora = _get_list(text_cfg, "corpora", where="text")
    target_datasets = text_cfg.get("target_datasets", None)
    target_set = None
    if isinstance(target_datasets, list) and target_datasets:
        target_set = {str(x).strip() for x in target_datasets if str(x).strip()}

    frames: List[pd.DataFrame] = []
    for c in corpora:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        path = c.get("path", None)

        if not name:
            continue
        if target_set is not None and name not in target_set:
            continue
        if path is None:
            continue

        p = Path(str(path)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"[text.corpora] File not found: {p} (dataset={name})")

        df = pd.read_json(p, lines=True)
        df = _normalize_schema(df, dataset_name=name)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No corpora loaded. Check text.corpora paths and text.target_datasets.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(by=["Dataset", "ID"], kind="mergesort").reset_index(drop=True)

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_json(combined_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved combined JSONL to: {combined_path} (n={len(merged)})")
    return str(combined_path)

def run_preprocess_chinese(config_path: Optional[str] = None, *, force_asr_jsonl: bool = False) -> str:
    cfg = load_text_config(config_path)
    text_cfg = _get_dict(cfg, "text", where="root")
    asr_cfg = _get_dict(cfg, "asr", where="root")

    # Step 0: (optional) rebuild NCMMSC JSONL from ASR CSV
    rebuild_ncmmsc_jsonl_from_asr_csv(asr_cfg=asr_cfg, text_cfg=text_cfg, force=force_asr_jsonl)

    # Step 1-2: merge corpora JSONL -> combined JSONL
    combined_jsonl = merge_corpora_to_jsonl(text_cfg=text_cfg)

    # Step 3-12: cleaning is in chinese_steps
    cleaned_jsonl = str(_require(text_cfg, "cleaned_jsonl", where="text"))
    df_clean = load_and_process_chinese(combined_jsonl, text_cfg=text_cfg)
    out_path = save_cleaned_jsonl(df_clean, output_path=cleaned_jsonl)
    return out_path

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge Chinese corpora and run paper-strict text preprocessing (YAML-driven).")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument(
        "--force-asr-jsonl",
        action="store_true",
        help="Rebuild text.ncmmsc_jsonl from asr.output_csv even if JSONL already exists.",
    )
    return p

def cli_main() -> None:
    ap = build_arg_parser()
    # IMPORTANT: ignore unknown args to be robust in notebook/colab contexts
    args, _unknown = ap.parse_known_args()

    out = run_preprocess_chinese(
        config_path=args.config,
        force_asr_jsonl=bool(args.force_asr_jsonl),
    )
    print(f"[INFO] DONE: {out}")

if __name__ == "__main__":
    cli_main()
