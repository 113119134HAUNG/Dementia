# preprocess_clear_text.py

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner  # 確認這個模組存在

# =============== 從 ASR CSV 轉成 NCMMSC JSONL ===============

ASR_CSV_PATH = "/content/ncmmsc_merged_asr_transcripts.csv"
NCMMSC_JSONL_PATH = "/content/data_Chinese-NCMMSC2021_AD_Competition.jsonl"

def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str):
    """將聲音轉文字的 CSV 轉成預處理流程會用到的 NCMMSC JSONL 格式。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    ncmmsc_df = pd.DataFrame({
        "ID": df["id"],
        "Diagnosis": df["label"],
        "Text_interviewer_participant": df["cleaned_transcript"].fillna(""),
        "Dataset": "NCMMSC2021_AD_Competition",
        "Languages": "zh",
    })

    ncmmsc_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved NCMMSC jsonl to: {jsonl_path}")


# =============== 通用：去掉英文、統一診斷標籤 ===============

def remove_english_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除 Languages == 'en' 的列（TAUKADIAL 英文部分）。"""
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]


# =============== 中文逐字稿清理函式 ===============

def clean_chinese_transcript(text: str) -> str:
    """
    溫和清理：
    - 去掉 Doctor:/Patient:/Speaker 1: 等說話者標籤
    - 去掉 .cha style 標記（時間戳、%mor、*PAR 等）
    - 去掉 [] () <> +code &code 類註記

    注意：不處理注音、不處理 [聽不清楚]，這些在 ASR 階段已經處理過。
    """
    if pd.isna(text):
        return ""

    t = str(text)

    # 先把全形空白之類換成正常空白
    t = t.replace("\u00A0", " ")

    # 說話者標籤
    t = re.sub(r"\b(Doctor|Patient|Interviewer)\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Speaker\s*\d+\s*[:：]", "", t, flags=re.IGNORECASE)
    # 若有中文：t = re.sub(r"(醫生|病人|受試者)\s*[:：]", "", t)

    # .cha 類型標記
    t = re.sub(r"\d+_\d+", "", t)             # 時間標記 30_5640
    t = re.sub(r"(%\w+|\*[A-Z]+):", "", t)    # %wor: *PAR:
    t = re.sub(r"\b\w+:\w+\|\w+", "", t)      # det:art|the n|scene

    # 各種括號與註記
    t = t.replace("‡", " ")
    t = re.sub(r"\[\+ *gram\]", " [gram] ", t)  # 保留 [+ gram]

    t = re.sub(r"\[[^\]]*\]", " ", t)          # 其他中括號
    t = re.sub(r"<[^>]*>", " ", t)             # <...>
    t = re.sub(r"\([^)]*\)", " ", t)           # (...)

    # 研究者 codes
    t = re.sub(r"\+<[^>]*>", " ", t)
    t = re.sub(r"\+[^ ]*", " ", t)
    t = re.sub(r"&\S+", " ", t)

    # 空白整理
    t = re.sub(r"\s+", " ", t).strip()

    return t


def clean_chinese_transcript_no_punct(text: str) -> str:
    """只留中英數字，不留標點的版本。"""
    t = clean_chinese_transcript(text)
    t = re.sub(r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]", " ", t)
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =============== 長度篩選（mean ± std） ===============

def filter_by_length(row, stats):
    mean = stats.loc[row["Diagnosis"], "mean"]
    std = stats.loc[row["Diagnosis"], "std"]
    return mean - std <= row["length"] <= mean + std


# =============== 主流程：包成工具函式 ===============

def run_preprocessing():
    # 1) CSV → NCMMSC JSONL
    csv_to_ncmmsc_jsonl(ASR_CSV_PATH, NCMMSC_JSONL_PATH)

    # 2) 合併多個 JSONL
    PREDICTIVE_JSONL_PATH = "/content/Chinese-predictive_challenge_tsv2_output.jsonl"
    TAUKADIAL_JSONL_PATH  = "/content/TAUKADIAL.jsonl"

    input_files = [
        NCMMSC_JSONL_PATH,
        PREDICTIVE_JSONL_PATH,
        TAUKADIAL_JSONL_PATH,
    ]

    output_directory = "/content/chinese_combined"
    os.makedirs(output_directory, exist_ok=True)

    output_filename = "Chinses_NCMMSC_iFlyTek_Taukdial.jsonl"

    combiner = JSONLCombiner(input_files, output_directory, output_filename)
    combiner.combine()
    print(f"[INFO] Combined jsonl saved to: {os.path.join(output_directory, output_filename)}")

    # 3) 讀合併後 JSONL
    combined_path = os.path.join(output_directory, output_filename)
    df_chinese = pd.read_json(combined_path, lines=True)

    # 看哪些沒有 Diagnosis 的列
    unknown_diagnosis_rows = df_chinese[df_chinese["Diagnosis"] == "Unknown"]
    print("Diagnosis == 'Unknown' 的 Dataset：", set(unknown_diagnosis_rows["Dataset"]))

    # 移除沒有標籤的資料
    df_chinese = df_chinese[df_chinese["Diagnosis"] != "Unknown"]

    # 統一標籤名稱
    df_chinese["Diagnosis"] = df_chinese["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})

    # 移除英文資料
    df_chinese = remove_english_rows(df_chinese)

    # 文字清理
    df_chinese["Text_interviewer_participant"] = (
        df_chinese["Text_interviewer_participant"].apply(clean_chinese_transcript)
    )

    # 長度統計
    df_chinese["length"] = df_chinese["Text_interviewer_participant"].apply(len)
    length_stats = df_chinese.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
    print("Length stats:\n", length_stats)

    # 過濾極端長度
    df_chinese = df_chinese[df_chinese.apply(filter_by_length, axis=1, stats=length_stats)]

    # train / test split
    train_cha, test_cha = train_test_split(
        df_chinese,
        test_size=0.2,
        stratify=df_chinese["Diagnosis"],
        random_state=42,
    )

    train_out = os.path.join(output_directory, "train_chinese.jsonl")
    test_out = os.path.join(output_directory, "test_chinese.jsonl")

    train_cha.to_json(train_out, orient="records", lines=True, force_ascii=False)
    test_cha.to_json(test_out, orient="records", lines=True, force_ascii=False)

    print("Saved train to:", train_out)
    print("Saved test to:", test_out)

if __name__ == "__main__":
    run_preprocessing()
