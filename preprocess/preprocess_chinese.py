# -*- coding: utf-8 -*-
"""
preprocess/preprocess_chinese.py (paper-strict, converged, CV-ready)

Pipeline:
1) ASR CSV -> NCMMSC JSONL (optional; controlled by flags)
2) Merge JSONLs using ONLY text.corpora (skip missing paths)
3) Dedup (Dataset, ID)
4) Normalize labels via ADType + YAML label_map
5) Language filter (YAML-driven)
6) Pre-clean subset (restrict dataset/labels only; NO balance/cap yet)
7) Clean text (tools.text_cleaning)
8) Prompt filter (ASR mixed speaker; YAML-driven; optional; before quality filter)
9) Quality filter (drop low-information / garbage text; YAML-driven, with drop-reason stats)
10) Final subset (apply_subset again to balance/cap on cleaned data)
11) Length filter (YAML-driven; auto-skip on small class size)
12) Save cleaned.jsonl (single set; no split)

NEW:
- Prompt filter stats logging (emptied/changed/avg len delta) for easier debugging.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from preprocess.collection import JSONLCombiner
from settings.dataset_subset import apply_subset
from settings.enums import ADType
from tools.config_utils import get_asr_config, get_text_config, load_text_config
from tools.text_cleaning import clean_asr_chinese, clean_structured_chinese, prompt_filter_text

UNK_TOKEN = "【聽不清楚】"

# =====================================================================
# helpers
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

def _get_list(cfg: Dict[str, Any], key: str, *, where: str = "") -> List[Any]:
    v = _require(cfg, key, where=where)
    if not isinstance(v, list):
        prefix = f"{where}." if where else ""
        raise ValueError(f"{prefix}{key} must be a list.")
    return v

def _resolve_path(p: Any) -> Path:
    return Path(str(p)).expanduser().resolve()

def _stable_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols: List[str] = []
    if "Dataset" in df.columns:
        cols.append("Dataset")
    if "ID" in df.columns:
        cols.append("ID")
    if cols:
        return df.sort_values(by=cols, kind="mergesort")
    if "ID" in df.columns:
        return df.sort_values(by="ID", kind="mergesort")
    return df.sort_index(kind="mergesort")

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
# Step 3-11: Load + dedup + normalize + filter + subset + clean + prompt + quality + length
# =====================================================================
def dedup_records(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "Dataset" in out.columns and "ID" in out.columns:
        dup = int(out.duplicated(subset=["Dataset", "ID"]).sum())
        if dup > 0:
            print(f"[WARN] Found {dup} duplicated rows by (Dataset, ID) -> keep first.")
            out = out.drop_duplicates(subset=["Dataset", "ID"], keep="first")
    return out.reset_index(drop=True)

def normalize_diagnosis_labels(df: pd.DataFrame, *, label_map: Dict[str, Any]) -> pd.DataFrame:
    if "Diagnosis" not in df.columns:
        raise ValueError("Expected column 'Diagnosis' not found.")

    mp_norm: Dict[str, str] = {}
    for k, v in (label_map or {}).items():
        k_norm = ADType._normalize(str(k))
        mp_norm[k_norm] = str(v).strip()

    def _map_one(x: Any) -> str:
        raw = "" if x is None else str(x).strip()
        raw_norm = ADType._normalize(raw)

        if raw_norm in mp_norm:
            raw = mp_norm[raw_norm]

        try:
            return ADType.from_any(raw).value
        except Exception:
            return "Unknown"

    out = df.copy()
    out["Diagnosis"] = out["Diagnosis"].apply(_map_one)

    unk_n = int((out["Diagnosis"] == "Unknown").sum())
    if unk_n > 0:
        print(f"[WARN] Diagnosis == 'Unknown' rows = {unk_n} (may be removed by apply_subset if target_labels set).")

    return out.reset_index(drop=True)

def drop_languages(df: pd.DataFrame, *, drop_langs: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "Languages" not in df.columns:
        return df
    dl = {str(x).strip().lower() for x in (drop_langs or []) if str(x).strip()}
    if not dl:
        return df
    lang = df["Languages"].astype(str).str.strip().str.lower()
    return df.loc[~lang.isin(dl)].reset_index(drop=True)

def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Text_interviewer_participant" not in df.columns:
        raise ValueError("Expected column 'Text_interviewer_participant' not found.")
    out = df.copy()
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].apply(clean_asr_chinese)
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].apply(clean_structured_chinese)
    return out

def _subset_cfg_preclean(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(text_cfg)
    cfg["balance"] = False
    cfg.pop("cap_per_class", None)
    return cfg

def quality_filter(
    df: pd.DataFrame,
    *,
    enabled: bool,
    min_chars: int,
    min_lex_chars: int,
    min_han: int,
    max_unk_ratio: float,
    unk_token: str = UNK_TOKEN,
) -> pd.DataFrame:
    if not enabled or df is None or df.empty:
        return df

    out = df.copy()
    s = out["Text_interviewer_participant"].fillna("").astype(str).str.strip()

    total_len = s.str.len()

    marker_pat = r"(\[//\]|\[/\]|\[\+\s*gram\]|\<\.\.\.\>|&-(?:uh|um))"
    lex = s.str.replace(unk_token, "", regex=False)
    lex = lex.str.replace(marker_pat, "", regex=True)
    lex_nos = lex.str.replace(r"\s+", "", regex=True)

    lex_len = lex_nos.str.len()
    han_cnt = lex.str.count(r"[\u4e00-\u9fff]")

    total_len_safe = total_len.clip(lower=1)
    unk_count = s.str.count(unk_token)
    unk_char_mass = unk_count * len(unk_token)
    unk_ratio = (unk_char_mass / total_len_safe)

    cond_chars = total_len >= int(min_chars)
    cond_lex = lex_len >= int(min_lex_chars)
    cond_han = han_cnt >= int(min_han)
    cond_unk = unk_ratio <= float(max_unk_ratio)

    keep = cond_chars & cond_lex & cond_han & cond_unk

    n0 = len(out)
    dropped = int((~keep).sum())
    print(f"[INFO] Quality filter: {n0} -> {n0 - dropped} (dropped {dropped})")
    print(
        "[INFO] Quality filter drop reasons (counts, may overlap): "
        f"short={int((~cond_chars).sum())}, "
        f"low_lex={int((~cond_lex).sum())}, "
        f"low_han={int((~cond_han).sum())}, "
        f"high_unk={int((~cond_unk).sum())}"
    )

    filtered = out.loc[keep].reset_index(drop=True)
    if not filtered.empty:
        print("[INFO] Label distribution after quality filter:\n", filtered["Diagnosis"].value_counts())
    return filtered

def length_filter(
    df: pd.DataFrame,
    *,
    enabled: bool,
    std_k: float,
    min_class_n: int = 30,
) -> pd.DataFrame:
    if not enabled or df is None or df.empty:
        return df
    if std_k < 0:
        raise ValueError("text.length_filter.std_k must be >= 0.")

    vc = df["Diagnosis"].value_counts()
    if (vc < int(min_class_n)).any():
        print(f"[INFO] Length filter skipped (min_class_n={min_class_n}, class_counts={vc.to_dict()}).")
        return df

    out = df.copy()
    out["length"] = out["Text_interviewer_participant"].apply(len)

    stats = out.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
    stats["std"] = stats["std"].fillna(0.0)
    print("[INFO] Length stats by Diagnosis:\n", stats)

    merged = out.merge(
        stats[["mean", "std"]],
        left_on="Diagnosis",
        right_index=True,
        how="left",
        suffixes=("", "_stat"),
    )

    lower = merged["mean"] - (float(std_k) * merged["std"])
    upper = merged["mean"] + (float(std_k) * merged["std"])
    keep = (merged["length"] >= lower) & (merged["length"] <= upper)

    filtered = out.loc[keep.values].reset_index(drop=True)
    filtered = filtered.drop(columns=["length"], errors="ignore")

    print(f"[INFO] After length filtering: {len(filtered)} samples remaining.")
    if not filtered.empty:
        print("[INFO] Label distribution after filtering:\n", filtered["Diagnosis"].value_counts())
    return filtered

def _log_prompt_filter_stats(before_s: pd.Series, after_s: pd.Series, *, scope_name: str) -> None:
    try:
        before_len = before_s.fillna("").astype(str).str.len()
        after_len = after_s.fillna("").astype(str).str.len()
        emptied = int((after_len == 0).sum())
        changed = int((before_s.fillna("").astype(str) != after_s.fillna("").astype(str)).sum())
        n = int(len(before_s))
        avg_delta = float((before_len - after_len).mean()) if n > 0 else 0.0
        print(f"[INFO] Prompt filter ({scope_name}): emptied={emptied}/{n}, changed={changed}/{n}, avg_len_delta={avg_delta:.2f}")
    except Exception:
        print(f"[WARN] Prompt filter stats failed for scope={scope_name} (non-fatal).")

def load_and_process_chinese(merged_jsonl_path: str, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    df = pd.read_json(Path(merged_jsonl_path), lines=True)

    df = dedup_records(df)

    label_map = text_cfg.get("label_map", {}) or {}
    df = normalize_diagnosis_labels(df, label_map=label_map)

    lang_cfg = _get_dict(text_cfg, "language_filter", where="text")
    df = drop_languages(df, drop_langs=lang_cfg.get("drop_languages", []) or [])

    # (A) pre-clean subset: restrict scope only
    df = apply_subset(df, _subset_cfg_preclean(text_cfg))
    print(f"[INFO] After pre-clean subset: {len(df)} samples remaining.")
    if not df.empty:
        print("[INFO] Label distribution after pre-clean subset:\n", df["Diagnosis"].value_counts())

    if not df.empty and (df["Diagnosis"].astype(str) == "Unknown").any():
        raise ValueError("Found Diagnosis=='Unknown' after pre-clean subset. Fix label_map/ADType or target_labels.")

    # (B) cleaning
    df = clean_text_column(df)
    if df.empty:
        return df.reset_index(drop=True)

    # (B2) prompt filter (YAML-driven; defaults safe)
    pf_cfg = text_cfg.get("prompt_filter", {})
    if not isinstance(pf_cfg, dict):
        pf_cfg = {}

    pf_enabled = bool(pf_cfg.get("enabled", False))
    pf_apply_datasets = pf_cfg.get("apply_datasets", None)
    if pf_apply_datasets is not None and not isinstance(pf_apply_datasets, list):
        pf_apply_datasets = None

    pf_mode = str(pf_cfg.get("mode", "leading"))
    pf_max_leading = int(pf_cfg.get("max_leading_sentences", 8))
    pf_min_keep = int(pf_cfg.get("min_keep_chars", 20))
    pf_patterns = pf_cfg.get("patterns", []) or []
    if not isinstance(pf_patterns, list):
        pf_patterns = []

    # ---- NEW: echo-only patterns (apply with mode="any" after leading) ----
    pf_echo_patterns = pf_cfg.get("echo_patterns", []) or []
    if not isinstance(pf_echo_patterns, list):
        pf_echo_patterns = []
    pf_echo_enabled = bool(pf_cfg.get("echo_enabled", True)) and (len(pf_echo_patterns) > 0)

    if pf_enabled and "Text_interviewer_participant" in df.columns:
        if pf_apply_datasets and "Dataset" in df.columns:
            apply_set = {str(x) for x in pf_apply_datasets}
            mask = df["Dataset"].astype(str).isin(apply_set)
            if mask.any():
                before = df.loc[mask, "Text_interviewer_participant"].copy()

                # Pass 1: leading (safe)
                after_leading = before.apply(
                    lambda x: prompt_filter_text(
                        x,
                        enabled=True,
                        patterns=pf_patterns,
                        mode=pf_mode,
                        max_leading_sentences=pf_max_leading,
                        min_keep_chars=pf_min_keep,
                    )
                )
                _log_prompt_filter_stats(before, after_leading, scope_name="prompt_leading(dataset_masked)")

                # Pass 2: any, echo-only (remove initial_prompt echoes anywhere)
                if pf_echo_enabled:
                    after_echo = after_leading.apply(
                        lambda x: prompt_filter_text(
                            x,
                            enabled=True,
                            patterns=pf_echo_patterns,
                            mode="any",
                            max_leading_sentences=0,   # unused in "any"
                            min_keep_chars=pf_min_keep,
                        )
                    )
                    _log_prompt_filter_stats(after_leading, after_echo, scope_name="prompt_echo_any(dataset_masked)")
                else:
                    after_echo = after_leading

                df.loc[mask, "Text_interviewer_participant"] = after_echo
        else:
            before = df["Text_interviewer_participant"].copy()

            after_leading = before.apply(
                lambda x: prompt_filter_text(
                    x,
                    enabled=True,
                    patterns=pf_patterns,
                    mode=pf_mode,
                    max_leading_sentences=pf_max_leading,
                    min_keep_chars=pf_min_keep,
                )
            )
            _log_prompt_filter_stats(before, after_leading, scope_name="prompt_leading(all_rows)")

            if pf_echo_enabled:
                after_echo = after_leading.apply(
                    lambda x: prompt_filter_text(
                        x,
                        enabled=True,
                        patterns=pf_echo_patterns,
                        mode="any",
                        max_leading_sentences=0,
                        min_keep_chars=pf_min_keep,
                    )
                )
                _log_prompt_filter_stats(after_leading, after_echo, scope_name="prompt_echo_any(all_rows)")
            else:
                after_echo = after_leading

            df["Text_interviewer_participant"] = after_echo

    # (C) quality filter
    qf_cfg = text_cfg.get("quality_filter", {})
    if not isinstance(qf_cfg, dict):
        qf_cfg = {}

    qf_enabled = bool(qf_cfg.get("enabled", True))
    min_chars = int(qf_cfg.get("min_chars", 15))
    min_lex_chars = int(qf_cfg.get("min_lex_chars", 8))
    min_han = int(qf_cfg.get("min_han", 6))
    max_unk_ratio = float(qf_cfg.get("max_unk_ratio", 0.75))

    df = quality_filter(
        df,
        enabled=qf_enabled,
        min_chars=min_chars,
        min_lex_chars=min_lex_chars,
        min_han=min_han,
        max_unk_ratio=max_unk_ratio,
        unk_token=UNK_TOKEN,
    )

    # (D) final subset after cleaning+filters
    df = apply_subset(df, text_cfg)
    print(f"[INFO] After final subset: {len(df)} samples remaining.")
    if not df.empty:
        print("[INFO] Label distribution after final subset:\n", df["Diagnosis"].value_counts())

    if not df.empty and (df["Diagnosis"].astype(str) == "Unknown").any():
        raise ValueError("Found Diagnosis=='Unknown' after final apply_subset. Fix label_map/ADType or target_labels.")

    # (E) length filter
    lf_cfg = _get_dict(text_cfg, "length_filter", where="text")
    enabled = bool(lf_cfg.get("enabled", True))
    std_k = float(_require(lf_cfg, "std_k", where="text.length_filter"))
    min_class_n = int(lf_cfg.get("min_class_n", 30))

    df = length_filter(df, enabled=enabled, std_k=std_k, min_class_n=min_class_n)

    df = _stable_sort(df).reset_index(drop=True)
    return df

# =====================================================================
# Save cleaned.jsonl (NO split)
# =====================================================================
def save_cleaned_jsonl(df: pd.DataFrame, *, output_path: str) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        raise ValueError("df is None (nothing to save).")

    df = _stable_sort(df).reset_index(drop=True)
    df.to_json(out_p, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved cleaned JSONL to: {out_p} (n={len(df)})")
    return str(out_p)

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

    # Step 3-11: process
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
