# text_cleaning.py

import re
import pandas as pd

# ====== 給 ASR 用的：處理注音、聽不清楚 等 ======

# 注音符號範圍，用來把 ㄚㄚㄚ 收斂成「嗯」
ZHUYIN_PATTERN = re.compile(
    r"[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒ"
    r"ㄓㄔㄕㄖㄗㄘㄙ"
    r"ㄧㄨㄩ"
    r"ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+"
)

def clean_asr_chinese(text: str) -> str:

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    t = str(text)

    # 把 [聽不清楚] 拿掉，避免影響後處理
    t = t.replace("[聽不清楚]", " ")

    # 連續注音 → 嗯
    t = ZHUYIN_PATTERN.sub("嗯", t)

    # 多個「嗯」連在一起 → 一個
    t = re.sub(r"(嗯[\s、，,.!?]*){2,}", "嗯 ", t)

    # 壓縮空白
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ====== 處理 Doctor:/%mor/註記 等 ======
def clean_structured_chinese(text: str) -> str:

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    t = str(text)

    # 先把全形空白換成一般空白
    t = t.replace("\u00A0", " ")

    # 說話者標籤：Doctor: / Patient: / Interviewer: / Speaker 1:
    t = re.sub(r"\b(Doctor|Patient|Interviewer)\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Speaker\s*\d+\s*[:：]", "", t, flags=re.IGNORECASE)
    
    # 中文標籤可以加
    t = re.sub(r"(醫生|病人|受試者)\s*[:：]", "", t)

    # .cha 類型標記
    t = re.sub(r"\d+_\d+", "", t)             # 時間標記 30_5640
    t = re.sub(r"(%\w+|\*[A-Z]+):", "", t)    # %wor: *PAR:
    t = re.sub(r"\b\w+:\w+\|\w+", "", t)      # det:art|the n|scene

    # 各種括號與註記
    t = t.replace("‡", " ")
    t = re.sub(r"\[\+ *gram\]", " [gram] ", t)  # 保留 [+ gram] 標準化

    t = re.sub(r"\[[^\]]*\]", " ", t)          # 其他中括號 [...]
    t = re.sub(r"<[^>]*>", " ", t)             # <...>
    t = re.sub(r"\([^)]*\)", " ", t)           # (...)

    # 研究者 codes
    t = re.sub(r"\+<[^>]*>", " ", t)   # +< ... >
    t = re.sub(r"\+[^ ]*", " ", t)     # 其他 +code
    t = re.sub(r"&\S+", " ", t)        # y&... 類型

    # 壓縮空白
    t = re.sub(r"\s+", " ", t).strip()
    return t

# structured 清理，拔掉中英文標點，只留中英數字 + 空白
def clean_structured_chinese_no_punct(text: str) -> str:
    t = clean_structured_chinese(text)
    t = re.sub(r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]", " ", t)
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
