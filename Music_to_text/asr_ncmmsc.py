# -*- coding: utf-8 -*-
"""
asr_ncmmsc.py

Run Whisper ASR on the NCMMSC2021 Chinese audio dataset:

    audio (AD / HC / MCI)
        → Whisper ASR
        → CSV (transcript + cleaned_transcript)

Configuration (single source of truth)
--------------------------------------
All paths and model settings are read from ``config_text.yaml`` (section ``asr``):

    asr:
      data_root      : root directory with subfolders AD / HC / MCI
      output_csv     : ASR CSV output path
      model_size     : Whisper model size (e.g. "large-v2")
      device         : "cuda" / "cpu"
      compute_type   : "float16", ...
      initial_prompt : Chinese clinical prompt
      decode         : dict of Whisper decoding hyper-parameters

This module is responsible for:
    - discovering audio files
    - running Whisper
    - writing ASR results to CSV (schema & cleaning via :mod:`asr_io`)
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from faster_whisper import WhisperModel

from asr_io import open_asr_csv_writer, write_asr_row_with_cleaning
from config_utils import get_asr_config

try:
    from tqdm.auto import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Folder name → diagnosis label (aligned with ADType.AD / HC / MCI)
LABEL_DIRS: Dict[str, str] = {
    "AD": "AD",
    "HC": "HC",
    "MCI": "MCI",
}

# Accepted audio extensions
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# =====================================================================
# Scan audio directory
# =====================================================================

def iter_audio_files(data_root: Path) -> Iterable[Tuple[str, str, str]]:
    """Yield (label, sample_id, audio_path) under data_root/AD,HC,MCI."""
    for label, subdir in LABEL_DIRS.items():
        audio_dir = data_root / subdir
        if not audio_dir.is_dir():
            print(f"[WARN] Missing directory, skip: {audio_dir}")
            continue

        for path in sorted(audio_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTS:
                continue

            sample_id = f"{label}_{path.stem}"
            yield label, sample_id, str(path)

# =====================================================================
# Build decode kwargs from YAML
# =====================================================================

def build_decode_kwargs(asr_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Construct Whisper decoding kwargs from ``asr.decode`` config block.

    All beam-search / temperature / threshold hyper-parameters live in YAML
    to keep experiments explicit and reproducible.

    Raises
    ------
    KeyError
        If the ``decode`` block is missing from the ``asr`` config.
    """
    decode_cfg = asr_cfg.get("decode")
    if decode_cfg is None:
        raise KeyError(
            "Config 檔缺少 'asr.decode' 區塊，"
            "請在 config_text.yaml 中加入 Whisper 解碼參數。"
        )
    # Return a shallow copy to avoid accidental in-place modification
    return dict(decode_cfg)

# =====================================================================
# Core ASR pipeline
# =====================================================================

def run_ncmmsc_asr(config_path: Optional[str] = None) -> None:
    """Run the NCMMSC ASR pipeline (audio → Whisper → CSV).

    Parameters
    ----------
    config_path : str or None
        Path to ``config_text.yaml``.
        If None, :func:`config_utils.load_text_config` will use its default.
    """
    # Load ASR config once
    asr_cfg = get_asr_config(path=config_path)

    data_root = Path(asr_cfg["data_root"])
    output_csv = Path(asr_cfg["output_csv"])
    model_size = asr_cfg["model_size"]
    device = asr_cfg["device"]
    compute_type = asr_cfg["compute_type"]
    initial_prompt = asr_cfg["initial_prompt"]

    # Load Whisper model
    print(f"[INFO] Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Collect audio files
    files = list(iter_audio_files(data_root))
    total_files = len(files)
    print(f"[INFO] Found {total_files} audio files under {data_root}")

    # Decode settings (fully YAML-driven)
    decode_kwargs = build_decode_kwargs(asr_cfg)
    decode_kwargs["initial_prompt"] = initial_prompt

    ok_count = 0
    err_count = 0

    # Open CSV writer (schema + light cleaning handled in asr_io)
    fp, writer = open_asr_csv_writer(output_csv)
    try:
        iterable = (
            tqdm(files, desc="Transcribing", unit="file", leave=True)
            if HAS_TQDM
            else files
        )

        for label, sample_id, audio_path in iterable:
            if HAS_TQDM:
                iterable.set_postfix_str(Path(audio_path).name)
            else:
                print(f"[INFO] Transcribing {audio_path} ...")

            try:
                segments, info = model.transcribe(audio_path, **decode_kwargs)
                raw_text = " ".join(
                    seg.text.strip()
                    for seg in segments
                    if seg.text.strip()
                )
                duration = getattr(info, "duration", 0.0)
                ok_count += 1
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] 無法轉錄 {audio_path}: {e}")
                raw_text = ""
                duration = 0.0
                err_count += 1

            write_asr_row_with_cleaning(
                writer,
                sample_id=sample_id,
                label=label,
                raw_transcript=raw_text,
                audio_path=audio_path,
                duration=duration,
            )
    finally:
        fp.close()

    print(
        f"\n[INFO] Done. Saved {total_files} rows to {output_csv} "
        f"(ok={ok_count}, error={err_count})"
    )

# =====================================================================
# CLI entry point
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ASR script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Whisper ASR on NCMMSC audio and export CSV "
            "(paths & hyper-parameters from config_text.yaml)."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config_text.yaml（預設：專案根目錄的 config_text.yaml）",
    )
    return parser

def cli_main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_ncmmsc_asr(config_path=args.config)

if __name__ == "__main__":
    cli_main()
