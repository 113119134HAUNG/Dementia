import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collection import JSONLCombiner  # 你原來的模組，如果實際檔名是 collections 要自己改


# =============== 1. 路徑與輸入檔 ===============

input_files = [
    "path/to/data_Chinese -NCMMSC2021_AD_Competition.jsonl",
    "path/to/Chinese-predictive challenge_tsv2_output.jsonl",
    "path/to/TAUKADIAL.jsonl"
]

output_directory = "path_to_output_directory"  # 末尾不要忘記加 / 或改用 os.path.join
output_filename = "combined_jsonl_Chinses_NCMMSC_iFlyTek_Taukdial.jsonl"

# 先把多個 jsonl 合併
combiner = JSONLCombiner(input_files, output_directory, output_filename)
combiner.combine()


# =============== 2. 通用：去掉英文、統一診斷標籤 ===============

def remove_english_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除 Languages == 'en' 的列（TAUKADIAL 英文部分）。"""
    if "Languages" not in df.columns:
        return df
    return df[df["Languages"] != "en"]


# =============== 3. 中文逐字稿清理函式 ===============

# 注音符號範圍，用來把 ㄚㄚㄚ 這種收斂成「嗯」
ZHUYIN_PATTERN = re.compile(r"[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+")


def clean_chinese_transcript(text: str) -> str:
    """
    適合中文會話資料（NCMMSC + Predictive Chinese + Taukdial）的溫和清理：
    - 保留中文、語氣詞
    - 去掉 .cha style 標記（時間戳、%mor、*PAR 等）
    - 去掉 Doctor:/Patient:/Speaker 1: 等說話者標籤
    - 移除各種 [] () <> +code &code 註記
    - 把一串注音符號收斂成一個「嗯」
    """
    if pd.isna(text):
        return ""

    t = str(text)

    # 先把全形空白之類換成正常空白
    t = t.replace("\u00A0", " ")

    # ===== 說話者標籤（不同資料集常見） =====
    # Doctor: / Patient: / Interviewer: / Speaker 1: 等
    t = re.sub(r"\b(Doctor|Patient|Interviewer)\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Speaker\s*\d+\s*[:：]", "", t, flags=re.IGNORECASE)
    # 如果你後面加上中文標籤，也可以一起處理，例如：
    # t = re.sub(r"(醫生|病人|受試者)\s*[:：]", "", t)

    # NCMMSC initial prompt 中可能有的自訂標記
    t = t.replace("[聽不清楚]", " ")

    # ===== .cha 類型標記 =====
    # 時間標記：30_5640 類
    t = re.sub(r"\d+_\d+", "", t)

    # 標頭：%wor: %mor: %gra: *PAR: *INV: ...
    t = re.sub(r"(%\w+|\*[A-Z]+):", "", t)

    # 形態標記：det:art|the n|scene 之類
    t = re.sub(r"\b\w+:\w+\|\w+", "", t)

    # ===== 各種括號/註記 =====
    # 特別保留 [+ gram] 這類，如果你真的需要，可在這裡調整
    t = t.replace("‡", " ")  # 特殊符號
    # 先把 [+ gram] 標準化（如果有的話）
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

    # ===== 注音 / 拉丁垃圾 =====
    # 一串注音 => 「嗯」
    t = ZHUYIN_PATTERN.sub("嗯", t)
    # 多個嗯連在一起 => 一個即可
    t = re.sub(r"(嗯[\s、，,.!?]*){2,}", "嗯 ", t)

    # ===== 空白與標點整理 =====
    # 你如果想保留中英文標點，這裡就只壓縮空白
    t = re.sub(r"\s+", " ", t)
    t = t.strip()

    return t


# 如果你還想要一個「給模型用的極簡版本」（例如拿去做 BoW / embedding），
# 可以再定義一個只留中英數字、不留標點的版本：
def clean_chinese_transcript_no_punct(text: str) -> str:
    t = clean_chinese_transcript(text)
    # 移除常見中英文標點
    t = re.sub(r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]", " ", t)
    # 只保留中文、英文、數字、空白
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# =============== 4. 篩長度（mean ± std） ===============

def filter_by_length(row, stats):
    mean = stats.loc[row["Diagnosis"], "mean"]
    std = stats.loc[row["Diagnosis"], "std"]
    return mean - std <= row["length"] <= mean + std


# =============== 5. 主處理流程 ===============

# 讀剛結合好的 jsonl
combined_path = os.path.join(output_directory, output_filename)
df_chinese = pd.read_json(combined_path, lines=True)

# 先看有哪些沒有 Diagnosis 的列（僅供檢查）
unknown_diagnosis_rows = df_chinese[df_chinese["Diagnosis"] == "Unknown"]
print("Diagnosis == 'Unknown' 的 Dataset：", set(unknown_diagnosis_rows["Dataset"]))

# 移除沒有標籤的資料（例如 iFlytek 測試集）
df_chinese = df_chinese[df_chinese["Diagnosis"] != "Unknown"]

# 統一診斷標籤名稱：NC/CTRL => HC
df_chinese["Diagnosis"] = df_chinese["Diagnosis"].replace({"NC": "HC", "CTRL": "HC"})

# 移除英文資料（TAUKADIAL 的英文）
df_chinese = remove_english_rows(df_chinese)

# 如果有 Predictive_Chinese_challenge_Chinese_2019，那一套會有 Doctor:/Patient:，
# 我們已經在 clean_chinese_transcript 裡處理，不一定要再分開寫。
# 但如果你有其他 dataset-specific 特殊規則，也可以在這裡補。

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