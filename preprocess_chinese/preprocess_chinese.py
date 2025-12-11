# -*- coding: utf-8 -*-
"""
preprocess_chinese.py

中文文字資料預處理 pipeline（文本層，不含聲學）：
    1. NCMMSC ASR CSV  →  NCMMSC 文字 JSONL（欄位對齊其他語料）
    2. 合併多個中文語料：
         - NCMMSC2021_AD_Competition（從 ASR 來）
         - Chinese-predictive_challenge
         - TAUKADIAL
    3. 統一標籤 + 移除英文資料
    4. 結構性文字清理（去掉 Doctor:/%mor/註記等）
    5. 依 Diagnosis 以長度 mean±std 過濾極端樣本
    6. stratified train/test split，輸出 JSONL

設定來源（single source of truth）
---------------------------------
全部路徑與檔名均由 `config_text.yaml` 的兩個區塊提供：

- asr:
    - output_csv          : NCMMSC ASR 輸出 CSV（步驟 1 的輸入）
- text:
    - ncmmsc_jsonl        : NCMMSC 轉出的中間 JSONL（步驟 1 的輸出）
    - predictive_jsonl    : Predictive 中文 JSONL
    - taukadial_jsonl     : TAUKADIAL JSONL
    - output_dir          : 合併 & 預處理後統一輸出資料夾
    - combined_name       : 合併全部語料的 JSONL 檔名
    - train_jsonl         : train split 的檔名（相對於 output_dir）
    - test_jsonl          : test  split 的檔名（相對於 output_dir）

使用方式（Example Usage）
-------------------------
In notebook:
    from preprocess_chinese import run_chinese_preprocessing

    # 使用預設 config_text.yaml
    run_chinese_preprocessing()

    # 或指定另一份 YAML（例如不同實驗設定）
    run_chinese_preprocessing(config_path="path/to/other_config.yaml")

From CLI:
    python preprocess_chinese.py
    python preprocess_chinese.py --config path/to/other_config.yaml
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from collection import JSONLCombiner
from text_cleaning import clean_structured_chinese
from config_utils import get_asr_config, get_text_config

# ===== ASR CSV → NCMMSC JSONL =====
def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str) -> str:
    """將 NCMMSC 的 ASR CSV 轉成 JSONL（欄位與其他中文語料對齊）。

    Input CSV schema (from ASR stage, see :mod:`asr_io`):
        - id
        - label
        - transcript
        - cleaned_transcript
        - audio_path
        - duration

    Output JSONL schema (for downstream text modeling):
        - ID                         : same as id
        - Diagnosis                  : same as label (標籤統一交給 enums.ADType 處理)
        - Text_interviewer_participant : cleaned_transcript（文字層級）
        - Dataset                    : 固定 "NCMMSC2021_AD_Competition"
        - Languages                  : 固定 "zh"
    """
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

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    ncmmsc_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved NCMMSC jsonl to: {jsonl_path}")
    return jsonl_path

# ===== 工具：移除英文、長度篩選 =====
def remove_english_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除 Languages == 'en' 的列（例如 TAUKADIAL 的英文部分）。"""
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]

def filter_by_length(row: pd.Series, stats: pd.DataFrame) -> bool:
    """依照 Diagnosis 的長度分布，做 mean±std 過濾極端樣本。"""
    mean = stats.loc[row["Diagnosis"], "mean"]
    std = stats.loc[row["Diagnosis"], "std"]
    return mean - std <= row["length"] <= mean + std

# ===== 合併 JSONL =====
def combine_jsonls(
    ncmmsc_jsonl: str,
    predictive_jsonl: str,
    taukadial_jsonl: str,
    output_dir: str,
    merged_name: str,
) -> str:
    """合併三個中文語料成一個 JSONL。

    Inputs
    ------
    ncmmsc_jsonl, predictive_jsonl, taukadial_jsonl : str
        三個語料的 JSONL 路徑。
    output_dir : str
        合併後輸出的資料夾。
    merged_name : str
        合併後 JSONL 檔名（例如 "Chinese_NCMMSC_iFlyTek_Taukdial.jsonl"）。
    """
    input_files = [ncmmsc_jsonl, predictive_jsonl, taukadial_jsonl]
    os.makedirs(output_dir, exist_ok=True)

    combiner = JSONLCombiner(input_files, output_dir, merged_name)
    combiner.combine()

    merged_path = os.path.join(output_dir, merged_name)
    print(f"[INFO] Combined jsonl saved to: {merged_path}")
    return merged_path

# ===== 讀取 + 清理 + 長度篩選 =====
def load_and_clean_chinese(merged_jsonl_path: str) -> pd.DataFrame:
    """讀取合併 JSONL，做標籤清理、語言過濾、結構性文字清理與長度過濾。"""
    df = pd.read_json(merged_jsonl_path, lines=True)

    # 找出沒有標籤的列
    unknown = df[df["Diagnosis"] == "Unknown"]
    if not unknown.empty:
        print("Diagnosis == 'Unknown' 的 Dataset：", set(unknown["Dataset"]))

    # 移除沒有 Diagnosis 的資料
    df = df[df["Diagnosis"] != "Unknown"]

    # 統一標籤名稱（NC / CTRL → HC）
    df["Diagnosis"] = df["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})

    # 移除英文資料列（只保留中文）
    df = remove_english_rows(df)

    # 結構性文字清理（去掉 Doctor:/%mor/註記等）
    df["Text_interviewer_participant"] = df["Text_interviewer_participant"].apply(
        clean_structured_chinese
    )

    # 計算長度並做 mean±std 過濾
    df["length"] = df["Text_interviewer_participant"].apply(len)
    length_stats = df.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
    print("Length stats:\n", length_stats)

    df = df[df.apply(filter_by_length, axis=1, stats=length_stats)]
    return df

# ===== split + 儲存 =====
def split_and_save(
    df: pd.DataFrame,
    output_dir: str,
    train_name: str,
    test_name: str,
):
    """做 stratified train/test split 並存成 JSONL。

    Parameters
    ----------
    df : pd.DataFrame
        清理後的完整資料表。
    output_dir : str
        輸出資料夾（會自動建立）。
    train_name, test_name : str
        train/test 的檔名（相對於 output_dir），通常由 YAML 的
        text.train_jsonl / text.test_jsonl 提供。
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Diagnosis"],
        random_state=42,
    )

    train_out = os.path.join(output_dir, train_name)
    test_out = os.path.join(output_dir, test_name)

    train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)

    print(f"[INFO] Saved train to: {train_out}")
    print(f"[INFO] Saved test  to: {test_out}")
    return train_out, test_out

# ===== 整體 pipeline =====
def run_chinese_preprocessing(config_path: Optional[str] = None):
    """主入口：從 config_text.yaml 讀取所有路徑，執行完整中文預處理流程。

    Parameters
    ----------
    config_path : str or None
        config_text.yaml 的路徑；若為 None，則由
        :func:`config_utils.load_text_config` 使用預設路徑。
    """
    # 從同一份 YAML 讀出 asr / text 兩個區塊
    asr_cfg = get_asr_config(path=config_path)
    text_cfg = get_text_config(path=config_path)

    # ASR CSV → NCMMSC JSONL
    asr_csv_path = asr_cfg["output_csv"]
    ncmmsc_jsonl_path = text_cfg["ncmmsc_jsonl"]

    csv_to_ncmmsc_jsonl(asr_csv_path, ncmmsc_jsonl_path)

    # 合併 NCMMSC + Predictive + TAUKADIAL
    merged_path = combine_jsonls(
        ncmmsc_jsonl=ncmmsc_jsonl_path,
        predictive_jsonl=text_cfg["predictive_jsonl"],
        taukadial_jsonl=text_cfg["taukadial_jsonl"],
        output_dir=text_cfg["output_dir"],
        merged_name=text_cfg["combined_name"],
    )

    # 清理 + 長度過濾
    df_clean = load_and_clean_chinese(merged_path)

    # train/test split + 存檔
    return split_and_save(
        df_clean,
        output_dir=text_cfg["output_dir"],
        train_name=text_cfg["train_jsonl"],
        test_name=text_cfg["test_jsonl"],
    )

# ===== CLI =====
def build_arg_parser() -> argparse.ArgumentParser:
    """極簡 CLI：只讓你指定 config 檔路徑，其餘全部交給 YAML 管理。"""
    parser = argparse.ArgumentParser(
        description="Preprocess Chinese AD datasets (NCMMSC + Predictive + TAUKADIAL) – config-driven."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="config_text.yaml 路徑（預設使用專案根目錄的 config_text.yaml）",
    )
    return parser

def cli_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_chinese_preprocessing(config_path=args.config)

if __name__ == "__main__":
    cli_main()
