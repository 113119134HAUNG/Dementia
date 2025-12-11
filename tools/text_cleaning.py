# -*- coding: utf-8 -*-
"""
text_cleaning.py

Text cleaning utilities used at two levels:

1. ASR-level cleaning (``clean_asr_chinese``):
   - Normalize Zhuyin sequences (ㄚㄚㄚ → 嗯)
   - Remove [聽不清楚] markers
   - Compress whitespace

2. Structured transcript cleaning (``clean_structured_chinese``):
   - Remove speaker labels (Doctor:, Patient:, *PAR:, etc.)
   - Strip CHAT-style annotations and brackets
   - Keep disfluency markers like [+ gram] normalized

3. A variant (``clean_structured_chinese_no_punct``) that additionally
   removes punctuation, leaving only Chinese characters, Latin letters,
   digits, and spaces.
"""

from __future__ import annotations

import re
from typing import Union

import pandas as pd

TextLike = Union[str, float]

# =====================================================================
# ASR-level cleaning
# =====================================================================

# Zhuyin code point range: collapse sequences to a single "嗯"
ZHUYIN_PATTERN = re.compile(
    r"[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒ"
    r"ㄓㄔㄕㄖㄗㄘㄙ"
    r"ㄧㄨㄩ"
    r"ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+"
)


def clean_asr_chinese(text: TextLike) -> str:
    """Light cleaning for Chinese ASR output."""

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    t = str(text)

    # Remove explicit [聽不清楚] markers
    t = t.replace("[聽不清楚]", " ")

    # Continuous Zhuyin sequences → 嗯
    t = ZHUYIN_PATTERN.sub("嗯", t)

    # Multiple 嗯 in a row → single 嗯
    t = re.sub(r"(嗯[\s、，,.!?]*){2,}", "嗯 ", t)

    # Compress whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

# =====================================================================
# Structured transcript cleaning
# =====================================================================

def clean_structured_chinese(text: TextLike) -> str:
    """Remove speaker tags, CHAT annotations, and miscellaneous codes,
    while keeping main Chinese/English text and disfluency markers.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    t = str(text)

    # Normalize non-breaking spaces
    t = t.replace("\u00A0", " ")

    # Speaker labels: Doctor: / Patient: / Interviewer: / Speaker 1:
    t = re.sub(r"\b(Doctor|Patient|Interviewer)\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Speaker\s*\d+\s*[:：]", "", t, flags=re.IGNORECASE)
    # Chinese speaker labels
    t = re.sub(r"(醫生|病人|受試者)\s*[:：]", "", t)

    # .cha-like markers
    t = re.sub(r"\d+_\d+", "", t)             # time codes 30_5640
    t = re.sub(r"(%\w+|\*[A-Z]+):", "", t)    # %wor: *PAR:
    t = re.sub(r"\b\w+:\w+\|\w+", "", t)      # det:art|the n|scene

    # Various brackets and annotations
    t = t.replace("‡", " ")
    # Normalize [+ gram] → [gram]
    t = re.sub(r"\[\+ *gram\]", " [gram] ", t)

    # Remove other bracketed content
    t = re.sub(r"\[[^\]]*\]", " ", t)   # [...]
    t = re.sub(r"<[^>]*>", " ", t)      # <...>
    t = re.sub(r"\([^)]*\)", " ", t)    # (...)

    # Researcher codes
    t = re.sub(r"\+<[^>]*>", " ", t)   # +< ... >
    t = re.sub(r"\+[^ ]*", " ", t)     # +code
    t = re.sub(r"&\S+", " ", t)        # y&... etc.

    # Compress whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_structured_chinese_no_punct(text: TextLike) -> str:
    """Structured cleaning + punctuation removal.

    Keep only Chinese characters, ASCII letters, digits, and spaces.
    """
    t = clean_structured_chinese(text)
    # Replace Chinese & English punctuation with spaces
    t = re.sub(
        r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]",
        " ",
        t,
    )
    # Remove remaining non [CJK / A-Z / a-z / 0-9 / space]
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
