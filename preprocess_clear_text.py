# preprocess_clear_text.py

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner

# =============== 從 ASR CSV 轉成 NCMMSC JSONL ===============

ASR_CSV_PATH = "/content/ncmmsc_merged_asr_transcripts.csv"
NCMMSC_JSONL_PATH = "/content/data_Chinese-NCMMSC2021_AD_Competition.jsonl"

def csv_to_ncmmsc_jsonl(csv_path: str, jsonl_path: str):
    """將聲音轉文字的 CSV 轉成預處理流程會用到的 NCMMSC JSONL 格式。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 建立符合預處理腳本預期的欄位：
    # - ID: 用原本的 id
    # - Diagnosis: 用 label (AD / MCI / HC)
    # - Text_interviewer_participant: 用 cleaned_transcript
    # - Dataset: 固定一個名稱
    # - Languages: 中文設 "zh"
    ncmmsc_df = pd.DataFrame({
        "ID": df["id"],
        "Diagnosis": df["label"],
        "Text_interviewer_participant": df["cleaned_transcript"].fillna(""),
        "Dataset": "NCMMSC2021_AD_Competition",
        "Languages": "zh",
    })

    ncmmsc_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"[INFO] Saved NCMMSC jsonl to: {jsonl_path}")


# =============== 路徑與輸入檔 ===============

# 先確保把 ASR CSV 轉成 NCMMSC JSONL
csv_to_ncmmsc_jsonl(ASR_CSV_PATH, NCMMSC_JSONL_PATH)

# TODO: 下面兩個路徑請換成你實際的 JSONL 路徑
PREDICTIVE_JSONL_PATH = "/content/Chinese-predictive_challenge_tsv2_output.jsonl"
TAUKADIAL_JSONL_PATH  = "/content/TAUKADIAL.jsonl"

input_files = [
    NCMMSC_JSONL_PATH,
    PREDICTIVE_JSONL_PATH,
    TAUKADIAL_JSONL_PATH,
]

# 輸出資料夾與檔名
output_directory = "/content/chinese_combined"
os.makedirs(output_directory, exist_ok=True)

output_filename = "Chinses_NCMMSC_iFlyTek_Taukdial.jsonl"

# 先把多個 jsonl 合併
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()
print(f"[INFO] Combined jsonl saved to: {os.path.join(output_directory, output_filename)}")

# =============== 通用：去掉英文、統一診斷標籤 ===============

def remove_english_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除 Languages == 'en' 的列（TAUKADIAL 英文部分）。"""
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]

# =============== 中文逐字稿清理函式 ===============

def clean_chinese_transcript(text: str) -> str:

    if pd.isna(text):
        return ""

    t = str(text)

    # 先把全形空白之類換成正常空白
    t = t.replace("\u00A0", " ")

    # ===== 說話者標籤（不同資料集常見） =====
    # Doctor: / Patient: / Interviewer: / Speaker 1: 等
    t = re.sub(r"\b(Doctor|Patient|Interviewer)\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Speaker\s*\d+\s*[:：]", "", t, flags=re.IGNORECASE)
    # 如果有中文標籤一起處理，例如：
    # t = re.sub(r"(醫生|病人|受試者)\s*[:：]", "", t)

    # ===== .cha 類型標記 =====
    # 時間標記：30_5640 類
    t = re.sub(r"\d+_\d+", "", t)

    # 標頭：%wor: %mor: %gra: *PAR: *INV: ...
    t = re.sub(r"(%\w+|\*[A-Z]+):", "", t)

    # 形態標記：det:art|the n|scene 之類
    t = re.sub(r"\b\w+:\w+\|\w+", "", t)

    # ===== 各種括號/註記 =====
    # 特別保留 [+ gram] 這類
    t = t.replace("‡", " ")  # 特殊符號
    # 把 [+ gram] 標準化
    t = re.sub(r"\[\+ *gram\]", " [gram] ", t)

    # 其他中括號內容全部拿掉
    t = re.sub(r"\[[^\]]*\]", " ", t)
    # 角括號內容 <...> 多半是修正 / 重說
    t = re.sub(r"<[^>]*>", " ", t)
    # 圓括號內容 (...) 多半是附註
    t = re.sub(r"\([^)]*\)", " ", t)

    # ===== 研究者 codes (+, &) =====
    t = re.sub(r"\+<[^>]*>", " ", t)   # +< ... >
    t = re.sub(r"\+[^ ]*", " ", t)     # 其他 +code
    t = re.sub(r"&\S+", " ", t)        # y&... 這種

    # （注音 → 嗯 的部分移除，因為已在 ASR clean_chinese_transcript 做過）

    # ===== 空白整理 =====
    t = re.sub(r"\s+", " ", t)
    t = t.strip()

    return t

def clean_chinese_transcript_no_punct(text: str) -> str:
    """
    只留中英數字、不留標點的版本（如果你之後要給模型用可以用這個）。
    """
    t = clean_chinese_transcript(text)
    t = re.sub(r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]", " ", t)
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# =============== 4. 篩長度（mean ± std） ===============

def filter_by_length(row, stats):
    mean = stats.loc[row["Diagnosis"], "mean"]
    std = stats.loc[row["Diagnosis"], "std"]
    return mean - std <= row["length"] <= mean + std

# =============== 5. 主處理流程 ===============

combined_path = os.path.join(output_directory, output_filename)
df_chinese = pd.read_json(combined_path, lines=True)

# 看有哪些沒有 Diagnosis 的列（僅供檢查）
unknown_diagnosis_rows = df_chinese[df_chinese["Diagnosis"] == "Unknown"]
print("Diagnosis == 'Unknown' 的 Dataset：", set(unknown_diagnosis_rows["Dataset"]))

# 移除沒有標籤的資料（例如 iFlytek 測試集）
df_chinese = df_chinese[df_chinese["Diagnosis"] != "Unknown"]

# 統一診斷標籤名稱：NC/CTRL => HC
df_chinese["Diagnosis"] = df_chinese["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})

# 移除英文資料（TAUKADIAL 的英文）
df_chinese = remove_english_rows(df_chinese)

# 套用中文清理到主要文本欄位
# 假設你的欄位仍然叫 "Text_interviewer_participant"
df_chinese["Text_interviewer_participant"] = df_chinese["Text_interviewer_participant"].apply(
    clean_chinese_transcript
)

# 文字長度（字元數）
df_chinese["length"] = df_chinese["Text_interviewer_participant"].apply(len)

# 各診斷別的長度統計
length_stats = df_chinese.groupby("Diagnosis")["length"].agg(["min", "max", "mean", "std"])
print("Length stats:\n", length_stats)

# 過濾掉太短/太長的異常值（在 mean ± std 範圍內的保留）
df_chinese = df_chinese[df_chinese.apply(filter_by_length, axis=1, stats=length_stats)]

# train / test 分割
train_cha, test_cha = train_test_split(
    df_chinese,
    test_size=0.2,
    stratify=df_chinese["Diagnosis"],
    random_state=42,
)

# 輸出為 jsonl，force_ascii=False 才會保留中文
train_out = os.path.join(output_directory, "train_chinese.jsonl")
test_out = os.path.join(output_directory, "test_chinese.jsonl")

train_cha.to_json(train_out, orient="records", lines=True, force_ascii=False)
test_cha.to_json(test_out, orient="records", lines=True, force_ascii=False)

print("Saved train to:", train_out)
print("Saved test to:", test_out)

