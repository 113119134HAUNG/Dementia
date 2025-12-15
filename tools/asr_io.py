# -*- coding: utf-8 -*-
"""
asr_io.py

I/O utilities for ASR outputs in this project.

Intentionally minimal:
- No CLI
- No config loading
- No printing
- No dependency on text_cleaning (caller injects cleaner if needed)

Defines:
    - canonical ASR CSV schema (column names)
    - helper to open a CSV writer with correct header
    - helper to build/write a single ASR result row (raw transcript always preserved)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple

# =====================================================================
# Canonical CSV schema
# =====================================================================
CSV_FIELDNAMES = (
    "id",
    "label",
    "transcript",
    "cleaned_transcript",
    "audio_path",
    "duration",
)

Cleaner = Callable[[str], str]

# =====================================================================
# File opening helper
# =====================================================================
def open_asr_csv_writer(output_csv: Path) -> Tuple[TextIO, csv.DictWriter]:
    """Open an ASR CSV file for writing and emit the header row."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fp = output_csv.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fp, fieldnames=list(CSV_FIELDNAMES))
    writer.writeheader()
    return fp, writer

# =====================================================================
# Row builder + writer
# =====================================================================
def build_asr_row(
    *,
    sample_id: Any,
    label: Any,
    raw_transcript: Any,
    audio_path: Any,
    duration: Any,
    cleaner: Optional[Cleaner] = None,
) -> Dict[str, Any]:
    """Build one ASR CSV row dict.

    Notes
    -----
    - `transcript` always stores the raw transcript (stringified).
    - `cleaned_transcript` is optional; only computed if `cleaner` is provided.
    """
    sid = "" if sample_id is None else str(sample_id).strip()
    lb = "" if label is None else str(label).strip()

    raw = "" if raw_transcript is None else str(raw_transcript)
    ap = "" if audio_path is None else str(audio_path)

    if cleaner is not None and raw:
        cleaned = cleaner(raw)
    else:
        cleaned = ""

    try:
        dur = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        dur = 0.0

    return {
        "id": sid,
        "label": lb,
        "transcript": raw,
        "cleaned_transcript": cleaned,
        "audio_path": ap,
        "duration": dur,
    }

def write_asr_row(
    writer: csv.DictWriter,
    *,
    sample_id: Any,
    label: Any,
    raw_transcript: Any,
    audio_path: Any,
    duration: Any,
    cleaner: Optional[Cleaner] = None,
) -> None:
    """Write a single ASR row to CSV (raw transcript preserved)."""
    row = build_asr_row(
        sample_id=sample_id,
        label=label,
        raw_transcript=raw_transcript,
        audio_path=audio_path,
        duration=duration,
        cleaner=cleaner,
    )
    writer.writerow(row)
