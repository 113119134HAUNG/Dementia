# -*- coding: utf-8 -*-
"""
asr_ncmmsc.py

NCMMSC2021 中文語音資料：
    音檔 (AD / HC / MCI) → Whisper ASR → CSV（含 transcript & cleaned_transcript）

設定來源：
    - 預設從 config_text.yaml 的 `asr` 區塊讀取
    - 也可以在函式參數或 CLI 參數中覆蓋設定

主要輸出：
    - 一個 CSV，欄位：
        id, label, transcript, cleaned_transcript, audio_path, duration
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

from faster_whisper import WhisperModel

from config_utils import get_asr_config
from text_cleaning import clean_asr_chinese  # 你自己的清理函式

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# 資料夾名稱 → 診斷標籤（與 ADType.HC / ADType.AD / ADType.MCI 一致）
LABEL_DIRS: Dict[str, str] = {
    "AD": "AD",
    "HC": "HC",
    "MCI": "MCI",
}

# 可接受的音訊副檔名
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


# ===== 掃資料夾，產生 (label, sample_id, audio_path) =====
def iter_audio_files(data_root: Path):
    """Yield (label, sample_id, audio_path) for all audio files under data_root/AD,HC,MCI."""
    for label, subdir in LABEL_DIRS.items():
        audio_dir = data_root / subdir
        if not audio_dir.is_dir():
            print(f"[WARN] 資料夾不存在，略過：{audio_dir}")
            continue

        for fname in sorted(os.listdir(audio_dir)):
            ext = Path(fname).suffix.lower()
            if ext not in AUDIO_EXTS:
                continue
            audio_path = audio_dir / fname
            sample_id = f"{label}_{audio_path.stem}"
            yield label, sample_id, str(audio_path)


def build_decode_kwargs(initial_prompt: str) -> Dict[str, Any]:
    """Whisper 解碼參數集中放這裡，之後要調整比較好管理。"""
    return dict(
        language="zh",
        task="transcribe",
        beam_size=10,
        patience=1.0,
        length_penalty=1.0,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        vad_filter=False,
        initial_prompt=initial_prompt,
        without_timestamps=True,
    )


# ===== 核心流程：給 notebook 呼叫 =====
def run_ncmmsc_asr(
    # 如果為 None，就從 config_text.yaml 讀
    data_root: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """
    NCMMSC 音檔 → Whisper ASR → 一個 CSV（含 transcript & cleaned_transcript）

    參數：
        data_root      : 音檔根目錄（底下有 AD / HC / MCI）
        output_csv     : 輸出 CSV 路徑
        model_size     : Whisper 模型大小 (e.g. "large-v2")
        device         : "cuda" 或 "cpu"
        compute_type   : "float16" 等
        initial_prompt : 中文提示詞
        config_path    : 自訂的 config_text.yaml 路徑（預設用專案根目錄的那份）
    """
    # 1) 讀設定檔
    asr_cfg = get_asr_config(path=config_path)

    # 2) 用「參數 > config」的優先順序
    data_root = Path(data_root or asr_cfg["data_root"])
    output_csv = Path(output_csv or asr_cfg["output_csv"])
    model_size = model_size or asr_cfg["model_size"]
    device = device or asr_cfg["device"]
    compute_type = compute_type or asr_cfg["compute_type"]
    initial_prompt = initial_prompt or asr_cfg["initial_prompt"]

    # 3) 載入模型
    print(f"[INFO] Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # 4) 準備檔案列表
    fieldnames = ["id", "label", "transcript", "cleaned_transcript", "audio_path", "duration"]
    files = list(iter_audio_files(data_root))
    total_files = len(files)
    print(f"[INFO] Found {total_files} audio files under {data_root}")

    decode_kwargs = build_decode_kwargs(initial_prompt)

    ok_count = 0
    err_count = 0

    # 5) 寫出 CSV
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        iterable = tqdm(files, desc="Transcribing", unit="file", leave=True) if HAS_TQDM else files

        for item in iterable:
            label, sample_id, audio_path = item

            if HAS_TQDM:
                iterable.set_postfix_str(Path(audio_path).name)
            else:
                print(f"[INFO] Transcribing {audio_path} ...")

            try:
                segments, info = model.transcribe(audio_path, **decode_kwargs)
                full_text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
                duration = getattr(info, "duration", 0.0)

                # 使用tools
                cleaned = clean_asr_chinese(full_text)

                ok_count += 1
            except Exception as e:
                print(f"[ERROR] 無法轉錄 {audio_path}: {e}")
                full_text = ""
                cleaned = ""
                duration = 0.0
                err_count += 1

            writer.writerow(
                {
                    "id": sample_id,
                    "label": label,
                    "transcript": full_text,
                    "cleaned_transcript": cleaned,
                    "audio_path": audio_path,
                    "duration": duration,
                }
            )

    print(
        f"\n[INFO] Done. Saved {total_files} rows to {output_csv} "
        f"(ok={ok_count}, error={err_count})"
    )


# ===== CLI 介面（可選） =====
def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser. Defaults are taken from config_text.yaml."""
    asr_cfg = get_asr_config()
    parser = argparse.ArgumentParser(description="Run Whisper ASR on NCMMSC audio and export CSV.")
    parser.add_argument("--config",type=str,default=None,help="config_text.yaml 路徑（預設使用專案根目錄的 config_text.yaml）",)
    parser.add_argument("--data-root",type=str,default=asr_cfg["data_root"],help="音檔根目錄（default: 由 config_text.yaml 讀取）",)
    parser.add_argument("--output-csv",type=str,default=asr_cfg["output_csv"],help="輸出 CSV 路徑（default: 由 config_text.yaml 讀取）",)
    parser.add_argument("--model-size",type=str,default=asr_cfg["model_size"],help="Whisper 模型大小（default: 由 config_text.yaml 讀取）",)
    parser.add_argument("--device",type=str,default=asr_cfg["device"],help='運算裝置："cuda" / "cpu"（default: 由 config_text.yaml 讀取）',)
    parser.add_argument("--compute-type",type=str,default=asr_cfg["compute_type"],help='計算型態："float16" 等（default: 由 config_text.yaml 讀取）',)
    return parser

def cli_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run_ncmmsc_asr(
        data_root=Path(args.data_root),
        output_csv=Path(args.output_csv),
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        config_path=args.config,
    )

if __name__ == "__main__":
    cli_main()
