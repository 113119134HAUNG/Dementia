# -*- coding: utf-8 -*-
"""
preprocess/chinese_steps.py

Holds Step 3-12:
- helpers
- dedup / label normalize / language filter
- cleaning
- TWO-STAGE prompt filter (leading + any for echo_patterns)
- quality filter
- final subset
- length filter
- save cleaned.jsonl
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from settings.dataset_subset import apply_subset
from settings.enums import ADType
from tools.text_cleaning import clean_asr_chinese, clean_structured_chinese, prompt_filter_text

UNK_TOKEN = "【聽不清楚】"

# =====================================================================
# helpers (shared by main + steps)
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

def _subset_cfg_preclean(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(text_cfg)
    cfg["balance"] = False
    cfg.pop("cap_per_class", None)
    return cfg

def _assert_no_unknown(df: pd.DataFrame, *, stage: str) -> None:
    if df is None or df.empty:
        return
    if (df["Diagnosis"].astype(str) == "Unknown").any():
        raise ValueError(f"Found Diagnosis=='Unknown' at stage={stage}. Fix label_map/ADType or target_labels.")

# =====================================================================
# Step 3: dedup / label normalize / language filter
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

# =====================================================================
# Step 7: cleaning
# =====================================================================
def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Text_interviewer_participant" not in df.columns:
        raise ValueError("Expected column 'Text_interviewer_participant' not found.")
    out = df.copy()
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].apply(clean_asr_chinese)
    out["Text_interviewer_participant"] = out["Text_interviewer_participant"].apply(clean_structured_chinese)
    return out

# =====================================================================
# Step 8: two-stage prompt filter (leading + any echo)
# =====================================================================
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

def _get_prompt_filter_cfg(text_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pf_cfg = text_cfg.get("prompt_filter", {})
    if not isinstance(pf_cfg, dict):
        pf_cfg = {}

    out = {
        "enabled": bool(pf_cfg.get("enabled", False)),
        "apply_datasets": pf_cfg.get("apply_datasets", None),
        "mode": str(pf_cfg.get("mode", "leading")),
        "max_leading_sentences": int(pf_cfg.get("max_leading_sentences", 8)),
        "min_keep_chars": int(pf_cfg.get("min_keep_chars", 20)),
        "patterns": pf_cfg.get("patterns", []) or [],
        "echo_enabled": bool(pf_cfg.get("echo_enabled", True)),
        "echo_patterns": pf_cfg.get("echo_patterns", []) or [],
    }

    if out["apply_datasets"] is not None and not isinstance(out["apply_datasets"], list):
        out["apply_datasets"] = None

    if not isinstance(out["patterns"], list):
        out["patterns"] = []
    if not isinstance(out["echo_patterns"], list):
        out["echo_patterns"] = []

    out["echo_enabled"] = bool(out["echo_enabled"]) and (len(out["echo_patterns"]) > 0)
    return out

def _apply_prompt_filter_two_stage_series(
    s: pd.Series,
    *,
    patterns: List[str],
    mode: str,
    max_leading_sentences: int,
    min_keep_chars: int,
    echo_enabled: bool,
    echo_patterns: List[str],
    scope_name: str,
) -> pd.Series:
    before = s.copy()

    # Pass 1: leading (safe)
    after_leading = before.apply(
        lambda x: prompt_filter_text(
            x,
            enabled=True,
            patterns=patterns,
            mode=mode,
            max_leading_sentences=max_leading_sentences,
            min_keep_chars=min_keep_chars,
        )
    )
    _log_prompt_filter_stats(before, after_leading, scope_name=f"prompt_leading({scope_name})")

    # Pass 2: any (echo-only)
    if echo_enabled:
        after_echo = after_leading.apply(
            lambda x: prompt_filter_text(
                x,
                enabled=True,
                patterns=echo_patterns,
                mode="any",
                max_leading_sentences=0,  # unused in any-mode
                min_keep_chars=min_keep_chars,
            )
        )
        _log_prompt_filter_stats(after_leading, after_echo, scope_name=f"prompt_echo_any({scope_name})")
        return after_echo

    return after_leading

def _apply_prompt_filter_two_stage(df: pd.DataFrame, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty or "Text_interviewer_participant" not in df.columns:
        return df

    pf = _get_prompt_filter_cfg(text_cfg)
    if not pf["enabled"]:
        return df

    out = df.copy()

    if pf["apply_datasets"] and "Dataset" in out.columns:
        apply_set = {str(x) for x in pf["apply_datasets"]}
        mask = out["Dataset"].astype(str).isin(apply_set)
        if mask.any():
            out.loc[mask, "Text_interviewer_participant"] = _apply_prompt_filter_two_stage_series(
                out.loc[mask, "Text_interviewer_participant"],
                patterns=pf["patterns"],
                mode=pf["mode"],
                max_leading_sentences=pf["max_leading_sentences"],
                min_keep_chars=pf["min_keep_chars"],
                echo_enabled=pf["echo_enabled"],
                echo_patterns=pf["echo_patterns"],
                scope_name="dataset_masked",
            )
        return out

    out["Text_interviewer_participant"] = _apply_prompt_filter_two_stage_series(
        out["Text_interviewer_participant"],
        patterns=pf["patterns"],
        mode=pf["mode"],
        max_leading_sentences=pf["max_leading_sentences"],
        min_keep_chars=pf["min_keep_chars"],
        echo_enabled=pf["echo_enabled"],
        echo_patterns=pf["echo_patterns"],
        scope_name="all_rows",
    )
    return out

# =====================================================================
# Step 9: quality filter
# =====================================================================
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

def _apply_quality_filter_from_cfg(df: pd.DataFrame, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    qf_cfg = text_cfg.get("quality_filter", {})
    if not isinstance(qf_cfg, dict):
        qf_cfg = {}

    return quality_filter(
        df,
        enabled=bool(qf_cfg.get("enabled", True)),
        min_chars=int(qf_cfg.get("min_chars", 15)),
        min_lex_chars=int(qf_cfg.get("min_lex_chars", 8)),
        min_han=int(qf_cfg.get("min_han", 6)),
        max_unk_ratio=float(qf_cfg.get("max_unk_ratio", 0.75)),
        unk_token=UNK_TOKEN,
    )

# =====================================================================
# Step 11: length filter
# =====================================================================
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

def _apply_length_filter_from_cfg(df: pd.DataFrame, text_cfg: Dict[str, Any]) -> pd.DataFrame:
    lf_cfg = _get_dict(text_cfg, "length_filter", where="text")
    enabled = bool(lf_cfg.get("enabled", True))
    std_k = float(_require(lf_cfg, "std_k", where="text.length_filter"))
    min_class_n = int(lf_cfg.get("min_class_n", 30))
    return length_filter(df, enabled=enabled, std_k=std_k, min_class_n=min_class_n)

# =====================================================================
# Step 3-12 main processing
# =====================================================================
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
    _assert_no_unknown(df, stage="pre_clean_subset")

    # (B) cleaning
    df = clean_text_column(df)
    if df.empty:
        return df.reset_index(drop=True)

    # (B2) two-stage prompt filter
    df = _apply_prompt_filter_two_stage(df, text_cfg)

    # (C) quality filter
    df = _apply_quality_filter_from_cfg(df, text_cfg)

    # (D) final subset after cleaning+filters
    df = apply_subset(df, text_cfg)
    print(f"[INFO] After final subset: {len(df)} samples remaining.")
    if not df.empty:
        print("[INFO] Label distribution after final subset:\n", df["Diagnosis"].value_counts())
    _assert_no_unknown(df, stage="final_subset")

    # (E) length filter
    df = _apply_length_filter_from_cfg(df, text_cfg)

    df = _stable_sort(df).reset_index(drop=True)
    return df

def save_cleaned_jsonl(df: pd.DataFrame, *, output_path: str) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        raise ValueError("df is None (nothing to save).")

    df = _stable_sort(df).reset_index(drop=True)
    df.to_json(out_p, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved cleaned JSONL to: {out_p} (n={len(df)})")
    return str(out_p)
