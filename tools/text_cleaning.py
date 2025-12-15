# -*- coding: utf-8 -*-
"""
text_cleaning.py

Text cleaning utilities (paper-aligned, strict + minimal).

Markers policy (paper-aligned)
------------------------------
- In CHAT-style *annotation markers*, keep ONLY:
    &-uh / &-um, [//], [/], <...>, [+ gram]
- All other annotation markers are removed.
- Lexical content (Chinese/English words) is preserved.

Additional rules
----------------
- Collapse consecutive duplicates of: <...>, [+ gram], &-uh/&-um
  (do NOT collapse [//] or [/]).
"""

from __future__ import annotations

import math
import numbers
import re
from typing import Union

TextLike = Union[str, float]

# =====================================================================
# Small helpers (no extra deps)
# =====================================================================
def _is_missing(x: TextLike) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return False
    if isinstance(x, numbers.Real):
        try:
            return math.isnan(float(x))
        except (TypeError, ValueError):
            return False
    return False

def _to_str(x: TextLike) -> str:
    return "" if _is_missing(x) else str(x)

# =====================================================================
# Shared regex
# =====================================================================
WS_PATTERN = re.compile(r"\s+")
BRACKET_ANY_PATTERN = re.compile(r"\[[^\]]*\]")

# =====================================================================
# ASR-level cleaning (light, deterministic)
# =====================================================================
ZHUYIN_PATTERN = re.compile(
    r"[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒ"
    r"ㄓㄔㄕㄖㄗㄘㄙ"
    r"ㄧㄨㄩ"
    r"ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+"
)
MULTI_HMM_PATTERN = re.compile(r"(嗯[\s、，,.!?]*){2,}")

def clean_asr_chinese(text: TextLike) -> str:
    t = _to_str(text)
    if not t:
        return ""
    t = t.replace("[聽不清楚]", " ")
    t = ZHUYIN_PATTERN.sub("嗯", t)
    t = MULTI_HMM_PATTERN.sub("嗯 ", t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t

# =====================================================================
# Structured transcript cleaning (paper-aligned, strict)
# =====================================================================
PAUSE_PATTERN = re.compile(r"<\s*(?:\.\s*){3,}>")  # < . . . > -> <...>
GRAM_PATTERN = re.compile(r"\[\+\s*gram\s*\]", flags=re.IGNORECASE)  # -> [+ gram]

# Normalize retrace/repeat variants (allow spaces inside)
RETRACE_PATTERN = re.compile(r"\[\s*//\s*\]")
REPEAT_PATTERN = re.compile(r"\[\s*/\s*\]")

KEEP_BRACKET_MARKERS = {"[//]", "[/]", "[+ gram]"}

SPEAKER_EN_PATTERN = re.compile(r"\b(Doctor|Patient|Interviewer)\s*[:：]", flags=re.IGNORECASE)
SPEAKER_NUM_PATTERN = re.compile(r"Speaker\s*\d+\s*[:：]", flags=re.IGNORECASE)
SPEAKER_ZH_PATTERN = re.compile(r"(醫生|病人|受試者)\s*[:：]")

TIME_CODE_PATTERN = re.compile(r"\d+_\d+")
MOR_PAR_PATTERN = re.compile(r"(%\w+|\*[A-Z]+):")
POS_TAG_PATTERN = re.compile(r"\b\w+:\w+\|\w+")

# STRICT: keep ONLY these filled pauses
ALLOWED_FILLPAUSES = {"uh", "um"}
FILLPAUSE_PATTERN = re.compile(r"(?i)&-([a-z]+)")
DROP_AMP_CODES_PATTERN = re.compile(r"&(?!(?:-[A-Za-z]+))\S+")

# Researcher codes
DROP_PLUS_ANGLE_PATTERN = re.compile(r"\+<[^>]*>")

# IMPORTANT: be conservative to avoid killing lexical forms like "C++"
# Only drop "+CODE" when it looks like a standalone annotation token:
#   - starts with '+' AND followed by letters/digits/underscore/hyphen
#   - bounded by whitespace or string edges
DROP_PLUS_CODE_PATTERN = re.compile(r"(?:(?<=\s)|^)\+[A-Za-z0-9_-]+(?=(?:\s|$))")

PAREN_PATTERN = re.compile(r"\([^)]*\)")
DROP_ANGLE_EXCEPT_PAUSE_PATTERN = re.compile(r"<(?!\.\.\.>)[^>]*>")  # keep literal <...>

# collapse consecutive duplicates (only these)
DUP_PAUSE_PATTERN = re.compile(r"(<\.\.\.>)(?:\s+\1)+")
DUP_GRAM_PATTERN = re.compile(r"(\[\+\s*gram\])(?:\s+\1)+", flags=re.IGNORECASE)
DUP_FILLPAUSE_PATTERN = re.compile(r"(&-(?:uh|um))(?:\s+\1)+", flags=re.IGNORECASE)

def _keep_allowed_fillpause(m: re.Match) -> str:
    w = m.group(1).lower()
    return f" &-{w} " if w in ALLOWED_FILLPAUSES else " "

def _collapse_marker_dups(t: str) -> str:
    t = DUP_PAUSE_PATTERN.sub(r"\1", t)
    t = DUP_GRAM_PATTERN.sub("[+ gram]", t)
    t = DUP_FILLPAUSE_PATTERN.sub(lambda m: m.group(1).lower(), t)
    return t

def clean_structured_chinese(text: TextLike) -> str:
    t = _to_str(text)
    if not t:
        return ""

    t = t.replace("\u00A0", " ")

    # Speaker labels
    t = SPEAKER_EN_PATTERN.sub("", t)
    t = SPEAKER_NUM_PATTERN.sub("", t)
    t = SPEAKER_ZH_PATTERN.sub("", t)

    # .cha-like markers
    t = TIME_CODE_PATTERN.sub("", t)
    t = MOR_PAR_PATTERN.sub("", t)
    t = POS_TAG_PATTERN.sub("", t)

    # Normalize key markers first
    t = PAUSE_PATTERN.sub(" <...> ", t)
    t = GRAM_PATTERN.sub(" [+ gram] ", t)

    t = RETRACE_PATTERN.sub(" [//] ", t)
    t = REPEAT_PATTERN.sub(" [/] ", t)

    # Filled pauses: keep only uh/um
    t = FILLPAUSE_PATTERN.sub(_keep_allowed_fillpause, t)

    # Remove other [...] but keep the 3 markers
    def _drop_other_brackets(m: re.Match) -> str:
        s = m.group(0)
        s_norm = WS_PATTERN.sub(" ", s.strip())
        return s_norm if s_norm in KEEP_BRACKET_MARKERS else " "

    t = BRACKET_ANY_PATTERN.sub(_drop_other_brackets, t)

    # Remove other <...> (except literal "<...>")
    t = DROP_ANGLE_EXCEPT_PAUSE_PATTERN.sub(" ", t)

    # Remove (...), misc symbols
    t = PAREN_PATTERN.sub(" ", t)
    t = t.replace("‡", " ")

    # Researcher codes
    t = DROP_PLUS_ANGLE_PATTERN.sub(" ", t)
    t = DROP_PLUS_CODE_PATTERN.sub(" ", t)

    # Remove &codes except &-word (we already normalized/dropped &-* above)
    t = DROP_AMP_CODES_PATTERN.sub(" ", t)

    # Final whitespace + appropriate collapse
    t = WS_PATTERN.sub(" ", t).strip()
    t = _collapse_marker_dups(t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t

# =====================================================================
# No punctuation variant
# =====================================================================
PUNCT_PATTERN = re.compile(r"[，。、「」『』？！：；（）《》〈〉——…,.!?;:()\"“”'\-]")
KEEP_BASIC_CHARS_PATTERN = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9_\s]")

DUP_PAUSE_TOK_PATTERN = re.compile(r"(PAUSE)(?:\s+\1)+")
DUP_GRAM_TOK_PATTERN = re.compile(r"(GRAM)(?:\s+\1)+")
DUP_FILLPAUSE_TOK_PATTERN = re.compile(r"(FILLPAUSE_(?:uh|um))(?:\s+\1)+", flags=re.IGNORECASE)

def _fillpause_to_token(m: re.Match) -> str:
    w = m.group(1).lower()
    return f" FILLPAUSE_{w} " if w in ALLOWED_FILLPAUSES else " "

def _collapse_token_dups(t: str) -> str:
    t = DUP_PAUSE_TOK_PATTERN.sub(r"\1", t)
    t = DUP_GRAM_TOK_PATTERN.sub(r"\1", t)
    t = DUP_FILLPAUSE_TOK_PATTERN.sub(lambda m: m.group(1).upper(), t)
    return t

def clean_structured_chinese_no_punct(text: TextLike) -> str:
    t = clean_structured_chinese(text)
    if not t:
        return ""

    t = t.replace("[//]", " RETRACE ")
    t = t.replace("[/]", " REPEAT ")
    t = t.replace("[+ gram]", " GRAM ")
    t = t.replace("<...>", " PAUSE ")
    t = FILLPAUSE_PATTERN.sub(_fillpause_to_token, t)

    t = PUNCT_PATTERN.sub(" ", t)
    t = KEEP_BASIC_CHARS_PATTERN.sub(" ", t)
    t = WS_PATTERN.sub(" ", t).strip()

    t = _collapse_token_dups(t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t
