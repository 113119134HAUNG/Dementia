# -*- coding: utf-8 -*-
"""
preprocess/preprocess_chinese.py (paper-strict, converged, CV-ready)

Pipeline:
1) ASR CSV -> NCMMSC JSONL (optional; controlled by flags)
2) Merge JSONLs using ONLY text.corpora (skip missing paths)
3-12) Done in preprocess/chinese_steps.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from preprocess.collection import JSONLCombiner
from tools.config_utils import get_asr_config, get_text_config, load_text_config

# --- support both: `python -m preprocess.preprocess_chinese` and direct run ---
try:
    from .chinese_steps import (
        _get_list,
        _require,
        _resolve_path,
        load_and_process_chinese,
        save_cleaned_jsonl,
    )
except Exception:  # pragma: no cover
    from chinese_steps import (  # type: ignore
        _get_list,
        _require,
        _resolve_path,
        load_and_process_chinese,
        save_cleaned_jsonl,
    )

# =====================================================================
# Step 1: ASR CSV -> NCMMSC JSONL
# =====================================================================
def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str, *, dataset_name: str) -> str:
    csv_path_p = Path(csv_path)
    jsonl_path_p = Path(jsonl_path)

    if not csv_path_p.exists():
        raise FileNotFoundError(f"ASR CSV not found: {csv_path_p} (run Music_to_text.asr_ncmmsc first)")

    df = pd.read_csv(csv_path_p, encoding="utf-8-sig")

    for col in ("id", "label", "transcript"):
        if col not in df.columns:
            raise ValueError(f"ASR CSV missing required column: {col} (paper-strict)")

    out_df = pd.DataFrame(
        {
            "ID": df["id"].astype(str),
            "Diagnosis": df["label"].astype(str),
            "Text_interviewer_participant": df["transcript"].fillna("").astype(str),
            "Dataset": str(dataset_name),
            "Languages": "zh",
        }
    )

    jsonl_path_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_json(jsonl_path_p, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved NCMMSC JSONL to: {jsonl_path_p} (n={len(out_df)})")
    return str(jsonl_path_p)

# =====================================================================
# Step 2: Merge JSONLs (only existing, ONLY text.corpora)
# =====================================================================
def combine_jsonls(*, corpora: List[Dict[str, Any]], output_dir: str, merged_name: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files: List[str] = []
    for c in corpora:
        if not isinstance(c, dict):
            raise ValueError("text.corpora entries must be dicts with keys: name, path.")
        if "name" not in c or "path" not in c:
            raise ValueError("Each text.corpora entry must contain keys: name, path.")

        path = c.get("path", None)
        if path is None:
            continue

        p = Path(str(path))
        if not p.exists():
            print(f"[WARN] Missing JSONL, skip: {p} (name={c.get('name')})")
            continue

        input_files.append(str(p))

    if not input_files:
        raise FileNotFoundError("No input JSONL files found for merging (check text.corpora paths).")

    print("[INFO] Combining JSONL files:")
    for f in input_files:
        print(f"  - {f}")

    combiner = JSONLCombiner(input_files, str(out_dir), merged_name)
    combiner.combine()

    merged_path = out_dir / merged_name
    print(f"[INFO] Combined JSONL saved to: {merged_path}")
    return str(merged_path)

# =====================================================================
# Orchestrator
# =====================================================================
def run_chinese_preprocessing(
    config_path: Optional[str] = None,
    *,
    skip_asr_jsonl: bool = False,
    force_asr_jsonl: bool = False,
) -> Tuple[str, str]:
    cfg = load_text_config(config_path)
    asr_cfg = get_asr_config(cfg=cfg)
    text_cfg = get_text_config(cfg=cfg)

    ncmmsc_jsonl = str(_require(text_cfg, "ncmmsc_jsonl", where="text"))
    output_dir = str(_require(text_cfg, "output_dir", where="text"))
    merged_name = str(_require(text_cfg, "combined_name", where="text"))
    cleaned_jsonl = str(text_cfg.get("cleaned_jsonl") or (Path(output_dir) / "cleaned.jsonl"))

    corpora = _get_list(text_cfg, "corpora", where="text")

    # paper-strict: corpora must contain an entry whose path == text.ncmmsc_jsonl
    ncmmsc_path = _resolve_path(ncmmsc_jsonl)
    ncmmsc_name = None
    for c in corpora:
        if isinstance(c, dict) and c.get("path") is not None:
            if _resolve_path(c["path"]) == ncmmsc_path:
                ncmmsc_name = c.get("name")
                break
    if not ncmmsc_name:
        raise ValueError("paper-strict: text.corpora must include NCMMSC entry matching text.ncmmsc_jsonl exactly.")

    # Step 1: ASR CSV -> NCMMSC JSONL (optional)
    ncmmsc_jsonl_p = Path(ncmmsc_jsonl)
    if skip_asr_jsonl:
        print(f"[INFO] Skip ASR CSV->JSONL. Using existing (if any): {ncmmsc_jsonl_p}")
    else:
        if ncmmsc_jsonl_p.exists() and not force_asr_jsonl:
            print(f"[INFO] NCMMSC JSONL exists, skip rebuild: {ncmmsc_jsonl_p} (use --force-asr-jsonl to rebuild)")
        else:
            asr_csv = str(_require(asr_cfg, "output_csv", where="asr"))
            csv_to_ncmmsc_jsonl(asr_csv, ncmmsc_jsonl, dataset_name=str(ncmmsc_name))

    # Step 2: merge JSONLs
    merged_path = combine_jsonls(corpora=corpora, output_dir=output_dir, merged_name=merged_name)

    # Step 3-11: process (in chinese_steps.py)
    df_clean = load_and_process_chinese(merged_path, text_cfg)

    # Step 12: save cleaned.jsonl
    cleaned_path = save_cleaned_jsonl(df_clean, output_path=cleaned_jsonl)

    return str(merged_path), str(cleaned_path)

# =====================================================================
# CLI
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess Chinese AD datasets (merge/clean) - paper-strict (no split).")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument("--skip-asr-jsonl", action="store_true", help="Skip ASR CSV -> NCMMSC JSONL conversion.")
    p.add_argument("--force-asr-jsonl", action="store_true", help="Force rebuild NCMMSC JSONL from ASR CSV.")
    return p

def cli_main() -> None:
    args = build_arg_parser().parse_args()
    run_chinese_preprocessing(
        config_path=args.config,
        skip_asr_jsonl=bool(args.skip_asr_jsonl),
        force_asr_jsonl=bool(args.force_asr_jsonl),
    )

if __name__ == "__main__":
    cli_main()
