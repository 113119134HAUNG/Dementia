# -*- coding: utf-8 -*-
"""
text_cleaning.py

Text cleaning utilities used at two levels:

1. ASR-level cleaning (clean_asr_chinese):
   - Normalize Zhuyin sequences (ㄚㄚㄚ → 嗯)
   - Remove [聽不清楚] markers
   - Compress whitespace

2. Structured transcript cleaning (clean_structured_chinese):
   - Remove speaker labels (Doctor:/%mor/ annotations, etc.)
   - Remove most CHAT-style annotations
   - BUT keep key disfluency / repair markers (paper-aligned):
       &-uh / &-um (filled pauses)
       [//]        (self-repair / retracing)
       [/]         (repetition)
       < . . . >   (pause; normalized to <...>)
       [+ gram]    (grammaticality marker; normalized to [+ gram])

3. A variant (clean_structured_chinese_no_punct) that additionally
   removes punctuation; disfluency markers are converted into
   alphabetic tokens before stripping.
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
# Structured transcript cleaning (paper-aligned: keep key markers)
# =====================================================================
# Normalize pause marker: < . . . > or <...> → <...>
PAUSE_PATTERN = re.compile(r"<\s*(?:\.\s*){3,}>")

# Normalize [+ gram] variants → [+ gram]
GRAM_PATTERN = re.compile(r"\[\+\s*gram\s*\]", flags=re.IGNORECASE)

# Keep these CHAT markers as-is (spacing normalized later)
KEEP_BRACKET_MARKERS = {"[//]", "[/]", "[+ gram]"}

def clean_structured_chinese(text: TextLike) -> str:
    """Remove speaker tags and most CHAT annotations, while keeping key
    disfluency / repair markers: &-uh, [//], [/], <...>, [+ gram].
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    t = str(text)

    # Normalize non-breaking spaces
    t = t.replace("\u00A0", " ")

    # -----------------------------------------------------------------
    # Speaker labels
    # -----------------------------------------------------------------
    t = re.sub(r"\b(Doctor|Patient|Interviewer)\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Speaker\s*\d+\s*[:：]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(醫生|病人|受試者)\s*[:：]", "", t)

    # -----------------------------------------------------------------
    # .cha-like markers
    # -----------------------------------------------------------------
    t = re.sub(r"\d+_\d+", "", t)                    # time codes 30_5640
    t = re.sub(r"(%\w+|\*[A-Z]+):", "", t)           # %wor: *PAR:
    t = re.sub(r"\b\w+:\w+\|\w+", "", t)             # det:art|the n|scene

    # -----------------------------------------------------------------
    # Keep / normalize key markers (paper-aligned)
    # -----------------------------------------------------------------
    # Pause: < . . . > → <...>
    t = PAUSE_PATTERN.sub(" <...> ", t)

    # [+ gram] normalization
    t = GRAM_PATTERN.sub(" [+ gram] ", t)

    # Ensure [//] and [/] have spacing if present
    t = t.replace("[//]", " [//] ")
    t = t.replace("[/]", " [/] ")

    # Filled pauses like &-uh / &-um: keep &-WORD, drop other &codes later
    t = re.sub(r"(?i)&-([a-z]+)", r" &-\1 ", t)

    # -----------------------------------------------------------------
    # Remove other bracketed / angled / parenthesized content
    # (but do NOT remove kept markers)
    # -----------------------------------------------------------------

    # Remove any [...] that is NOT one of: [//], [/], [+ gram]
    # (works after we normalized spacing above)
    def _drop_other_brackets(m: re.Match) -> str:
        s = m.group(0)
        s_norm = re.sub(r"\s+", " ", s.strip())
        return s_norm if s_norm in KEEP_BRACKET_MARKERS else " "

    t = re.sub(r"\[[^\]]*\]", _drop_other_brackets, t)

    # Remove any remaining <...> content except the pause token <...>
    # (we already normalized pauses to literal "<...>")
    t = re.sub(r"<(?!\.\.\.>)[^>]*>", " ", t)

    # Remove (...) content
    t = re.sub(r"\([^)]*\)", " ", t)

    # Misc symbols
    t = t.replace("‡", " ")

    # Researcher codes: remove +<...>, +code
    t = re.sub(r"\+<[^>]*>", " ", t)
    t = re.sub(r"\+[^ ]+", " ", t)

    # Remove &codes except &-word (filled pauses)
    t = re.sub(r"&(?!(?:-[A-Za-z]+))\S+", " ", t)

    # -----------------------------------------------------------------
    # Final whitespace cleanup
    # -----------------------------------------------------------------
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_structured_chinese_no_punct(text: TextLike) -> str:
    """Structured cleaning + punctuation removal.

    Keep only Chinese characters, ASCII letters, digits, and spaces.
    To preserve disfluency information after stripping symbols,
    we convert key markers into alphabetic tokens first.
    """
    t = clean_structured_chinese(text)

    # Convert key markers to alphabetic tokens (so they survive stripping)
    t = t.replace("[//]", " RETRACE ")
    t = t.replace("[/]", " REPEAT ")
    t = t.replace("[+ gram]", " GRAM ")
    t = t.replace("<...>", " PAUSE ")
    t = re.sub(r"(?i)&-([a-z]+)", r" FILLPAUSE_\1 ", t)

    # Replace Chinese & English punctuation with spaces
    t = re.sub(
        r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]",
        " ",
        t,
    )
    # Remove remaining non [CJK / A-Z / a-z / 0-9 / space / underscore]
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
