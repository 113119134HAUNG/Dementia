# -*- coding: utf-8 -*-
"""
asr_io.py
I/O utilities for ASR outputs in this project.
This module defines:
- The canonical ASR CSV schema (column names)
- Helper to open a CSV writer with the correct header
- Helper to write a single ASR result row, including text cleaning
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO, Tuple

from text_cleaning import clean_asr_chinese  # ASR 層級的輕量清理（注音→嗯、[聽不清楚] 等）

# ===== 公開的 CSV schema =====

#: Canonical field names for the ASR CSV used throughout the project.
#: 一筆 ASR 結果包含：
#:   - id                 : 唯一識別（例如 AD_0001）
#:   - label              : 診斷標籤（AD / HC / MCI）
#:   - transcript         : Whisper 原始逐字稿（未清理）
#:   - cleaned_transcript : 過濾後版本（供下游文字分析）
#:   - audio_path         : 原始音檔路徑
#:   - duration           : 音檔長度（秒）
CSV_FIELDNAMES = [
    "id",
    "label",
    "transcript",
    "cleaned_transcript",
    "audio_path",
    "duration",
]

# ===== 開檔工具 =====
def open_asr_csv_writer(output_csv: Path) -> Tuple[TextIO, csv.DictWriter]:
    """Open an ASR CSV file for writing and emit the header row.

    Parameters
    ----------
    output_csv : Path
        Output CSV path. Parent directories will be created if needed.

    Returns
    -------
    fp : TextIO
        The opened file handle (remember to close it after use).
    writer : csv.DictWriter
        Writer configured with :data:`CSV_FIELDNAMES` and header already written.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fp = output_csv.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    return fp, writer

# ===== 單筆結果寫入（含清理） =====
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

    The cleaning logic is delegated to :func:`text_cleaning.clean_asr_chinese`,
    so that this function acts as a thin adapter between the ASR model output
    and the canonical CSV schema.

    Parameters
    ----------
    writer : csv.DictWriter
        Writer returned by :func:`open_asr_csv_writer`.
    sample_id : str
        Unique identifier for this recording (e.g. "AD_0001").
    label : str
        Diagnosis label ("AD", "HC", "MCI", ...). Higher-level normalization
        is handled elsewhere via :class:`enums.ADType`.
    raw_transcript : str
        Raw text returned by the ASR model (concatenated segments).
    audio_path : str
        Absolute or relative path to the original audio file.
    duration : float
        Duration of the audio, in seconds.
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
