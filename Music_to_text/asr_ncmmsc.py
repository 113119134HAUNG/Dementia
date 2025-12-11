# -*- coding: utf-8 -*-
"""
asr_ncmmsc.py

NCMMSC2021 中文語音資料：
    音檔 (AD / HC / MCI)
        → Whisper ASR
        → CSV（含 transcript & cleaned_transcript）

Configuration (single source of truth)
--------------------------------------
All paths and model settings are taken from ``config_text.yaml`` (section ``asr``):

    - data_root      : root directory with subfolders AD / HC / MCI
    - output_csv     : where the ASR CSV will be written
    - model_size     : Whisper model size (e.g. "large-v2")
    - device         : "cuda" / "cpu"
    - compute_type   : "float16", ...
    - initial_prompt : Chinese clinical prompt
    - decode         : dict of Whisper decoding hyperparameters

Separation of Concerns
----------------------
- This module is responsible **only** for:
    - discovering audio files
    - running Whisper
    - counting successes / failures

- CSV schema, cleaning, and writing are delegated to :mod:`asr_io`.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional

from faster_whisper import WhisperModel

from config_utils import get_asr_config
from asr_io import open_asr_csv_writer, write_asr_row_with_cleaning

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# 資料夾名稱 → 診斷標籤（與 ADType.AD / ADType.HC / ADType.MCI 一致）
LABEL_DIRS: Dict[str, str] = {
    "AD": "AD",
    "HC": "HC",
    "MCI": "MCI",
}

# 可接受的音訊副檔名
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# ===== 掃資料夾，產生 (label, sample_id, audio_path) =====
def iter_audio_files(data_root: Path) -> Iterable[Tuple[str, str, str]]:
    """Yield (label, sample_id, audio_path) for all audio files under data_root/AD,HC,MCI."""
    for label, subdir in LABEL_DIRS.items():
        audio_dir = data_root / subdir
        if not audio_dir.is_dir():
            print(f"[WARN] 資料夾不存在，略過：{audio_dir}")
            continue

        for path in sorted(audio_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTS:
                continue

            sample_id = f"{label}_{path.stem}"
            yield label, sample_id, str(path)

# ===== Whisper decode 參數：完全由 YAML 驅動 =====
def build_decode_kwargs(asr_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Construct Whisper decoding kwargs from the ``asr.decode`` config block.

    All beam-search / temperature / threshold hyperparameters are expected
    to live in the YAML config, not in code, to keep the paper-style
    experimental setup explicit and reproducible.

    Raises
    ------
    KeyError
        If the ``decode`` block is missing from the ``asr`` config.
    """
    decode_cfg = asr_cfg.get("decode")
    if decode_cfg is None:
        raise KeyError("Config 檔缺少 'asr.decode' 區塊，請在 config_text.yaml 中加入。")
    # 回傳一份獨立 dict，避免之後被就地修改
    return dict(decode_cfg)

# ===== 核心流程：給 notebook / script 呼叫 =====
def run_ncmmsc_asr(config_path: Optional[str] = None) -> None:
    """Run the NCMMSC ASR pipeline (audio → Whisper → CSV).

    Parameters
    ----------
    config_path : str or None
        Path to ``config_text.yaml``.
        If None, :func:`config_utils.load_text_config` will use its default.
    """
    # 讀設定檔（只讀一次）
    asr_cfg = get_asr_config(path=config_path)

    data_root = Path(asr_cfg["data_root"])
    output_csv = Path(asr_cfg["output_csv"])
    model_size = asr_cfg["model_size"]
    device = asr_cfg["device"]
    compute_type = asr_cfg["compute_type"]
    initial_prompt = asr_cfg["initial_prompt"]

    # 載入 Whisper 模型
    print(f"[INFO] Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # 收集所有音檔
    files = list(iter_audio_files(data_root))
    total_files = len(files)
    print(f"[INFO] Found {total_files} audio files under {data_root}")

    #  Decode 參數（完全由 YAML 驅動）
    decode_kwargs = build_decode_kwargs(asr_cfg)
    # 保證 initial_prompt 一定有帶進去
    decode_kwargs["initial_prompt"] = initial_prompt

    ok_count = 0
    err_count = 0

    #  寫出 CSV（由 asr_io 處理 schema + 清理）
    fp, writer = open_asr_csv_writer(output_csv)
    try:
        iterable = tqdm(files, desc="Transcribing", unit="file", leave=True) if HAS_TQDM else files

        for label, sample_id, audio_path in iterable:
            if HAS_TQDM:
                iterable.set_postfix_str(Path(audio_path).name)
            else:
                print(f"[INFO] Transcribing {audio_path} ...")

            try:
                segments, info = model.transcribe(audio_path, **decode_kwargs)
                raw_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
                duration = getattr(info, "duration", 0.0)

                ok_count += 1
            except Exception as e:
                print(f"[ERROR] 無法轉錄 {audio_path}: {e}")
                raw_text = ""
                duration = 0.0
                err_count += 1

            # 寫入 CSV：清理 + 欄位 mapping 都在 asr_io 裡做
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

# ===== 極簡 CLI：只允許指定 config 檔 =====
def build_arg_parser() -> argparse.ArgumentParser:
    """極簡 CLI：只讓你指定 config 檔路徑，其餘全部交給 YAML 管理。"""
    parser = argparse.ArgumentParser(description="Run Whisper ASR on NCMMSC audio and export CSV (config-driven).")
    parser.add_argument("--config",type=str,default=None,help="config_text.yaml 路徑（預設使用專案根目錄的 config_text.yaml）",)
    return parser

def cli_main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_ncmmsc_asr(config_path=args.config)

if __name__ == "__main__":
    cli_main()
