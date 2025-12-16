# -*- coding: utf-8 -*-
"""
tools/text_cleaning.py

Text cleaning utilities (paper-aligned, strict + minimal).

NEW in this revision:
- Prompt filter sentence splitter more robust for ASR that lacks 。！？:
  include ， 、 , as weak sentence boundaries to avoid "whole paragraph becomes one sentence" over-drop.
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
# Unintelligible token normalization
# =====================================================================
UNK_TOKEN = "【聽不清楚】","【上海话】"
UNK_ANY_PATTERN = re.compile(r"[\[\【]\s*聽不清楚\s*[\]\】]")
UNK_DUP_PATTERN = re.compile(rf"(?:{re.escape(UNK_TOKEN)}\s*){{2,}}")

def _normalize_unk(t: str) -> str:
    t = UNK_ANY_PATTERN.sub(UNK_TOKEN, t)
    t = UNK_DUP_PATTERN.sub(f"{UNK_TOKEN} ", t)
    return t

# =====================================================================
# Weird unicode sanitizer (allowlist)
# =====================================================================
_ALLOWED_CHARS_PATTERN = re.compile(
    r"[^\u4e00-\u9fffA-Za-z0-9\s"
    r"\[\]\(\)<>"
    r"&\+\-_/\\"          # keep some marker symbols
    r"，。！？；：、（）「」『』《》…【】"
    r"\.,!?;:\"“”'·]+"
)

def sanitize_weird_unicode(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00A0", " ")
    t = _ALLOWED_CHARS_PATTERN.sub(" ", t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t

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

    t = _normalize_unk(t)

    t = ZHUYIN_PATTERN.sub("嗯", t)
    t = MULTI_HMM_PATTERN.sub("嗯 ", t)

    t = sanitize_weird_unicode(t)

    t = _normalize_unk(t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t

# =====================================================================
# Structured transcript cleaning (paper-aligned, strict)
# =====================================================================
PAUSE_PATTERN = re.compile(r"<\s*(?:\.\s*){3,}>")  # < . . . > -> <...>
GRAM_PATTERN = re.compile(r"\[\+\s*gram\s*\]", flags=re.IGNORECASE)  # -> [+ gram]

RETRACE_PATTERN = re.compile(r"\[\s*//\s*\]")
REPEAT_PATTERN = re.compile(r"\[\s*/\s*\]")

KEEP_BRACKET_MARKERS = {"[//]", "[/]", "[+ gram]"}

SPEAKER_EN_PATTERN = re.compile(r"\b(Doctor|Patient|Interviewer)\s*[:：]", flags=re.IGNORECASE)
SPEAKER_NUM_PATTERN = re.compile(r"Speaker\s*\d+\s*[:：]", flags=re.IGNORECASE)
SPEAKER_ZH_PATTERN = re.compile(r"(醫生|病人|受試者)\s*[:：]")

TIME_CODE_PATTERN = re.compile(r"\d+_\d+")
MOR_PAR_PATTERN = re.compile(r"(%\w+|\*[A-Z]+):")
POS_TAG_PATTERN = re.compile(r"\b\w+:\w+\|\w+")

ALLOWED_FILLPAUSES = {"uh", "um"}
FILLPAUSE_PATTERN = re.compile(r"(?i)&-([a-z]+)")

AMP_CJK_PATTERN = re.compile(r"&([\u4e00-\u9fff])")
DROP_AMP_CODES_PATTERN = re.compile(r"&(?!(?:-[A-Za-z]+))\S+")

DROP_PLUS_ANGLE_PATTERN = re.compile(r"\+<[^>]*>")
DROP_PLUS_CODE_PATTERN = re.compile(r"(?:(?<=\s)|^)\+[A-Za-z0-9_-]+(?=(?:\s|$))")

PAREN_PATTERN = re.compile(r"\([^)]*\)")
DROP_ANGLE_EXCEPT_PAUSE_PATTERN = re.compile(r"<(?!\.\.\.>)[^>]*>")

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
    t = _normalize_unk(t)

    t = SPEAKER_EN_PATTERN.sub("", t)
    t = SPEAKER_NUM_PATTERN.sub("", t)
    t = SPEAKER_ZH_PATTERN.sub("", t)

    t = TIME_CODE_PATTERN.sub("", t)
    t = MOR_PAR_PATTERN.sub("", t)
    t = POS_TAG_PATTERN.sub("", t)

    t = PAUSE_PATTERN.sub(" <...> ", t)
    t = GRAM_PATTERN.sub(" [+ gram] ", t)

    t = RETRACE_PATTERN.sub(" [//] ", t)
    t = REPEAT_PATTERN.sub(" [/] ", t)

    t = FILLPAUSE_PATTERN.sub(_keep_allowed_fillpause, t)
    t = AMP_CJK_PATTERN.sub(r"\1", t)

    def _drop_other_brackets(m: re.Match) -> str:
        s = m.group(0)
        s_norm = WS_PATTERN.sub(" ", s.strip())
        return s_norm if s_norm in KEEP_BRACKET_MARKERS else " "

    t = BRACKET_ANY_PATTERN.sub(_drop_other_brackets, t)
    t = DROP_ANGLE_EXCEPT_PAUSE_PATTERN.sub(" ", t)

    t = PAREN_PATTERN.sub(" ", t)
    t = t.replace("‡", " ")

    t = DROP_PLUS_ANGLE_PATTERN.sub(" ", t)
    t = DROP_PLUS_CODE_PATTERN.sub(" ", t)

    t = DROP_AMP_CODES_PATTERN.sub(" ", t)

    t = sanitize_weird_unicode(t)

    t = WS_PATTERN.sub(" ", t).strip()
    t = _collapse_marker_dups(t)
    t = _normalize_unk(t)
    t = WS_PATTERN.sub(" ", t).strip()
    return t

# =====================================================================
# Prompt filter (ASR mixed speaker)
# =====================================================================
# NOTE: add ，、, to avoid whole-paragraph as single sentence when ASR lacks 。！？.
SENT_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;，,、\n])\s*")

def prompt_filter_text(
    text: TextLike,
    *,
    enabled: bool = True,
    patterns: list[str] | None = None,
    mode: str = "leading",              # leading | any
    max_leading_sentences: int = 8,
    min_keep_chars: int = 20,
) -> str:
    """
    Remove interviewer prompts from ASR-mixed transcripts.

    - mode="leading": drop consecutive prompt-like sentences at start only (safe default)
    - mode="any": drop any sentence that matches (riskier)

    If output becomes too short, returns "" (let quality_filter drop it).
    """
    t = _to_str(text)
    if not enabled or not t:
        return t.strip()

    t = WS_PATTERN.sub(" ", t).strip()
    if not t:
        return ""

    pats = patterns or []
    regs: list[re.Pattern] = []
    for p in pats:
        try:
            regs.append(re.compile(p))
        except re.error:
            continue

    if not regs:
        return t

    parts = [s.strip() for s in SENT_SPLIT_PATTERN.split(t) if s.strip()]
    if not parts:
        return ""

    def _is_prompt_sent(s: str) -> bool:
        return any(rg.search(s) for rg in regs)

    mode_l = (mode or "leading").strip().lower()
    if mode_l == "any":
        kept = [s for s in parts if not _is_prompt_sent(s)]
    else:
        kept: list[str] = []
        dropped = 0
        lim = max(0, int(max_leading_sentences))
        for s in parts:
            if dropped < lim and _is_prompt_sent(s):
                dropped += 1
                continue
            kept.append(s)

    out = " ".join(kept).strip()
    out = WS_PATTERN.sub(" ", out).strip()

    if len(out) < int(min_keep_chars):
        return ""
    return out

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
