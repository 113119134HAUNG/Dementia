# -*- coding: utf-8 -*-
"""
asr_io.py

I/O utilities for ASR outputs in this project.

Defines:
    - the canonical ASR CSV schema (column names)
    - helper to open a CSV writer with the correct header
    - helper to write a single ASR result row, including light cleaning
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO, Tuple

from text_cleaning import clean_asr_chinese

# =====================================================================
# Canonical CSV schema
# =====================================================================

#: ASR CSV columns:
#:   - id                 : unique ID (e.g. "AD_0001")
#:   - label              : diagnosis label (AD / HC / MCI)
#:   - transcript         : raw Whisper transcript
#:   - cleaned_transcript : lightly cleaned transcript (for downstream NLP)
#:   - audio_path         : original audio path
#:   - duration           : audio length (seconds)
CSV_FIELDNAMES = [
    "id",
    "label",
    "transcript",
    "cleaned_transcript",
    "audio_path",
    "duration",
]

# =====================================================================
# File opening helper
# =====================================================================

def open_asr_csv_writer(output_csv: Path) -> Tuple[TextIO, csv.DictWriter]:
    """Open an ASR CSV file for writing and emit the header row."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fp = output_csv.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    return fp, writer

# =====================================================================
# Write single result row (with cleaning)
# =====================================================================

def write_asr_row_with_cleaning(
    writer: csv.DictWriter,
    *,
    sample_id: str,
    label: str,
    raw_transcript: str,
    audio_path: str,
    duration: float,
) -> None:
    """Write a single ASR result into the CSV, including transcript cleaning.

    Cleaning logic is delegated to :func:`text_cleaning.clean_asr_chinese`.
    """
    cleaned = clean_asr_chinese(raw_transcript)

    writer.writerow(
        {
            "id": sample_id,
            "label": label,
            "transcript": raw_transcript,
            "cleaned_transcript": cleaned,
            "audio_path": audio_path,
            "duration": duration,
        }
    )
