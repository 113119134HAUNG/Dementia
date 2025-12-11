# preprocess_chinese.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from collection import JSONLCombiner  # 你原來的模組
from text_cleaning import clean_structured_chinese

# ===== 預設路徑 =====

ASR_CSV_PATH = "/content/ncmmsc_merged_asr_transcripts.csv"
NCMMSC_JSONL_PATH = "/content/data_Chinese-NCMMSC2021_AD_Competition.jsonl"

PREDICTIVE_JSONL_PATH = "/content/Chinese-predictive_challenge_tsv2_output.jsonl"
TAUKADIAL_JSONL_PATH = "/content/TAUKADIAL.jsonl"

DEFAULT_OUTPUT_DIR = "/content/chinese_combined"
MERGED_JSONL_NAME = "combined_chinese_corpora.jsonl"


# ===== ASR CSV → NCMMSC JSONL =====

def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str) -> str:
    """將 NCMMSC 的 ASR CSV 轉成 JSONL（欄位與其他資料集統一）"""
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
    """移除 Languages == 'en' 的列（TAUKADIAL 英文部分）"""
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]


def filter_by_length(row, stats):
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
    input_files = [ncmmsc_jsonl, predictive_jsonl, taukadial_jsonl]
    os.makedirs(output_dir, exist_ok=True)

    combiner = JSONLCombiner(input_files, output_dir, merged_name)
    combiner.combine()

    merged_path = os.path.join(output_dir, merged_name)
    print(f"[INFO] Combined jsonl saved to: {merged_path}")
    return merged_path


# =====讀取 + 清理 + 長度篩選 =====
def load_and_clean_chinese(merged_jsonl_path: str) -> pd.DataFrame:
    df = pd.read_json(merged_jsonl_path, lines=True)

    # 看有哪些沒有 Diagnosis 的列
    unknown = df[df["Diagnosis"] == "Unknown"]
    if not unknown.empty:
        print("Diagnosis == 'Unknown' 的 Dataset：", set(unknown["Dataset"]))

    # 移除沒有標籤的資料
    df = df[df["Diagnosis"] != "Unknown"]

    # 統一標籤名稱
    df["Diagnosis"] = df["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})

    # 移除英文
    df = remove_english_rows(df)

    # 文字清理（結構性的、非聲學的）
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
def split_and_save(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Diagnosis"],
        random_state=42,
    )

    train_out = os.path.join(out_dir, "train_chinese.jsonl")
    test_out = os.path.join(out_dir, "test_chinese.jsonl")

    train_df.to_json(train_out, orient="records", lines=True, force_ascii=False)
    test_df.to_json(test_out, orient="records", lines=True, force_ascii=False)

    print(f"[INFO] Saved train to: {train_out}")
    print(f"[INFO] Saved test  to: {test_out}")
    return train_out, test_out


# ===== pipeline =====
def run_chinese_preprocessing(
    asr_csv_path: str = ASR_CSV_PATH,
    ncmmsc_jsonl_path: str = NCMMSC_JSONL_PATH,
    predictive_jsonl: str = PREDICTIVE_JSONL_PATH,
    taukadial_jsonl: str = TAUKADIAL_JSONL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    merged_name: str = MERGED_JSONL_NAME,
):
    # ASR CSV → NCMMSC JSONL
    csv_to_ncmmsc_jsonl(asr_csv_path, ncmmsc_jsonl_path)

    # 合併 NCMMSC + Predictive + TAUKADIAL
    merged_path = combine_jsonls(
        ncmmsc_jsonl=ncmmsc_jsonl_path,
        predictive_jsonl=predictive_jsonl,
        taukadial_jsonl=taukadial_jsonl,
        output_dir=output_dir,
        merged_name=merged_name,
    )

    # 清理 + 長度過濾
    df_clean = load_and_clean_chinese(merged_path)

    # train/test split + 存檔
    return split_and_save(df_clean, output_dir)


# ===== CLI =====
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess Chinese AD datasets (NCMMSC + Predictive + TAUKADIAL)."
    )
    parser.add_argument("--asr-csv", type=str, default=ASR_CSV_PATH, help="NCMMSC ASR CSV 路徑")
    parser.add_argument(
        "--ncmmsc-jsonl", type=str, default=NCMMSC_JSONL_PATH, help="中間 NCMMSC JSONL 路徑"
    )
    parser.add_argument(
        "--predictive-jsonl",
        type=str,
        default=PREDICTIVE_JSONL_PATH,
        help="Predictive Chinese jsonl 路徑",
    )
    parser.add_argument(
        "--taukadial-jsonl",
        type=str,
        default=TAUKADIAL_JSONL_PATH,
        help="TAUKADIAL jsonl 路徑",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="輸出資料夾")
    parser.add_argument("--merged-name", type=str, default=MERGED_JSONL_NAME, help="合併後 jsonl 檔名")
    return parser

def cli_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_chinese_preprocessing(
        asr_csv_path=args.asr_csv,
        ncmmsc_jsonl_path=args.ncmmsc_jsonl,
        predictive_jsonl=args.predictive_jsonl,
        taukadial_jsonl=args.taukadial_jsonl,
        output_dir=args.output_dir,
        merged_name=args.merged_name,
    )

if __name__ == "__main__":
    cli_main()
