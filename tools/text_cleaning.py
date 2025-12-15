# -*- coding: utf-8 -*-
"""
text_cleaning.py

Text cleaning utilities (paper-aligned, strict + minimal).

1) ASR-level cleaning (clean_asr_chinese):
   - Normalize Zhuyin sequences → "嗯"
   - Remove [聽不清楚]
   - Compress whitespace

2) Structured transcript cleaning (clean_structured_chinese):
   - Remove speaker labels and most CHAT-style annotations
   - Keep key disfluency / repair markers (paper-aligned):
       &-uh / &-um (filled pauses)          -> keep as &-uh style
       [//]        (self-repair / retrace)  -> keep
       [/]         (repetition)             -> keep
       < . . . >   (pause)                  -> normalize to <...>
       [+ gram]    (gram marker)            -> normalize to [+ gram]

3) No-punct variant (clean_structured_chinese_no_punct):
   - Convert kept markers to alphabetic tokens then strip punctuation/symbols
"""

from __future__ import annotations

import re
from typing import Union

TextLike = Union[str, float]

# =====================================================================
# Small helpers (no extra deps)
# =====================================================================
def _is_missing(x: TextLike) -> bool:
    # Treat None / float NaN as missing. (Pandas reads CSV missing as float NaN)
    return x is None or (isinstance(x, float) and x != x)  # NaN != NaN


def _to_str(x: TextLike) -> str:
    return "" if _is_missing(x) else str(x)

# =====================================================================
# Shared regex
# =====================================================================
WS_PATTERN = re.compile(r"\s+")
BRACKET_ANY_PATTERN = re.compile(r"\[[^\]]*\]")

# =====================================================================
# ASR-level cleaning
# =====================================================================
ZHUYIN_PATTERN = re.compile(
    r"[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒ"
    r"ㄓㄔㄕㄖㄗㄘㄙ"
    r"ㄧㄨㄩ"
    r"ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+"
)

MULTI_HMM_PATTERN = re.compile(r"(嗯[\s、，,.!?]*){2,}")

def clean_asr_chinese(text: TextLike) -> str:
    """Light cleaning for Chinese ASR output."""
    t = _to_str(text)
    if not t:
        return ""

    t = t.replace("[聽不清楚]", " ")
    t = ZHUYIN_PATTERN.sub("嗯", t)
    t = MULTI_HMM_PATTERN.sub("嗯 ", t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t

# =====================================================================
# Structured transcript cleaning (paper-aligned)
# =====================================================================
PAUSE_PATTERN = re.compile(r"<\s*(?:\.\s*){3,}>")  # < . . . > -> <...>
GRAM_PATTERN = re.compile(r"\[\+\s*gram\s*\]", flags=re.IGNORECASE)  # -> [+ gram]

KEEP_BRACKET_MARKERS = {"[//]", "[/]", "[+ gram]"}

SPEAKER_EN_PATTERN = re.compile(r"\b(Doctor|Patient|Interviewer)\s*[:：]", flags=re.IGNORECASE)
SPEAKER_NUM_PATTERN = re.compile(r"Speaker\s*\d+\s*[:：]", flags=re.IGNORECASE)
SPEAKER_ZH_PATTERN = re.compile(r"(醫生|病人|受試者)\s*[:：]")

TIME_CODE_PATTERN = re.compile(r"\d+_\d+")
MOR_PAR_PATTERN = re.compile(r"(%\w+|\*[A-Z]+):")
POS_TAG_PATTERN = re.compile(r"\b\w+:\w+\|\w+")

FILLPAUSE_PATTERN = re.compile(r"(?i)&-([a-z]+)")                 # keep only &-letters
DROP_AMP_CODES_PATTERN = re.compile(r"&(?!(?:-[A-Za-z]+))\S+")    # drop other &codes

DROP_PLUS_ANGLE_PATTERN = re.compile(r"\+<[^>]*>")
DROP_PLUS_CODE_PATTERN = re.compile(r"\+[^ ]+")

PAREN_PATTERN = re.compile(r"\([^)]*\)")
DROP_ANGLE_EXCEPT_PAUSE_PATTERN = re.compile(r"<(?!\.\.\.>)[^>]*>")  # keep literal <...>

def clean_structured_chinese(text: TextLike) -> str:
    """Remove speaker tags and most CHAT annotations, while keeping key markers."""
    t = _to_str(text)
    if not t:
        return ""

    t = t.replace("\u00A0", " ")  # NBSP -> space

    # Speaker labels
    t = SPEAKER_EN_PATTERN.sub("", t)
    t = SPEAKER_NUM_PATTERN.sub("", t)
    t = SPEAKER_ZH_PATTERN.sub("", t)

    # .cha-like markers
    t = TIME_CODE_PATTERN.sub("", t)
    t = MOR_PAR_PATTERN.sub("", t)
    t = POS_TAG_PATTERN.sub("", t)

    # Keep / normalize key markers
    t = PAUSE_PATTERN.sub(" <...> ", t)
    t = GRAM_PATTERN.sub(" [+ gram] ", t)
    t = t.replace("[//]", " [//] ")
    t = t.replace("[/]", " [/] ")
    t = FILLPAUSE_PATTERN.sub(r" &-\1 ", t)

    # Remove other [...] but keep the 3 markers above
    def _drop_other_brackets(m: re.Match) -> str:
        s = m.group(0)
        s_norm = WS_PATTERN.sub(" ", s.strip())
        return s_norm if s_norm in KEEP_BRACKET_MARKERS else " "

    t = BRACKET_ANY_PATTERN.sub(_drop_other_brackets, t)

    # Remove other <...> (except literal "<...>")
    t = DROP_ANGLE_EXCEPT_PAUSE_PATTERN.sub(" ", t)

    # Remove (...)
    t = PAREN_PATTERN.sub(" ", t)

    # Misc symbols
    t = t.replace("‡", " ")

    # Researcher codes
    t = DROP_PLUS_ANGLE_PATTERN.sub(" ", t)
    t = DROP_PLUS_CODE_PATTERN.sub(" ", t)

    # Remove &codes except &-word
    t = DROP_AMP_CODES_PATTERN.sub(" ", t)

    # Final whitespace
    t = WS_PATTERN.sub(" ", t).strip()
    return t

# =====================================================================
# No punctuation variant
# =====================================================================
PUNCT_PATTERN = re.compile(r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]")
KEEP_BASIC_CHARS_PATTERN = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_\s]")

def clean_structured_chinese_no_punct(text: TextLike) -> str:
    """Structured cleaning + punctuation removal.

    Preserve key disfluency info by converting markers to alphabetic tokens
    before stripping symbols.
    """
    t = clean_structured_chinese(text)
    if not t:
        return ""

    # Convert key markers to tokens
    t = t.replace("[//]", " RETRACE ")
    t = t.replace("[/]", " REPEAT ")
    t = t.replace("[+ gram]", " GRAM ")
    t = t.replace("<...>", " PAUSE ")
    t = FILLPAUSE_PATTERN.sub(r" FILLPAUSE_\1 ", t)

    # Strip punctuation/symbols
    t = PUNCT_PATTERN.sub(" ", t)
    t = KEEP_BASIC_CHARS_PATTERN.sub(" ", t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t
