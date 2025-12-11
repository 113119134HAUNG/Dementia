# asr_ncmmsc.py

import os
import csv
from pathlib import Path
import argparse

from faster_whisper import WhisperModel
from text_cleaning import clean_asr_chinese

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ========= 預設設定 =========
DEFAULT_DATA_ROOT = Path("/content/NCMMSC2021_AD_Competition-dev/dataset/merge")
DEFAULT_OUTPUT_CSV = Path("/content/ncmmsc_merged_asr_transcripts.csv")

DEFAULT_MODEL_SIZE = "large-v2"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

DEFAULT_INITIAL_PROMPT = (
    "以下是一段中文口語錄音的逐字稿，內容為針對失智症、阿茲海默症、"
    "輕度認知障礙（MCI）與健康對照組（HC）的臨床訪談與語言測驗，"
    "包括圖畫描述任務、語意流暢度測驗以及日常生活相關問答。"
    "請以繁體中文完整逐字轉寫受試者與訪談者的口語內容，不要翻譯、"
    "不要潤飾或做任何摘要，也不要自行更改語序或補上沒聽到的字。"
    "請保留口語語氣詞與猶豫聲（例如：嗯、呃、啊、就是、然後）、"
    "重複、修正、語法不完整或中斷的句子，這些特徵可能與認知功能相關。"
    "遇到下列專有名詞或相關用語時，請盡量使用常見且一致的寫法："
    "「失智症」、「阿茲海默症」、「阿茲海默病」、「輕度認知障礙」、"
    "「MCI」、「AD」、「HC」、「記憶力」、「認知功能」、"
    "「量表」、「測驗」、「醫師」、「受試者」、「照顧者」。"
    "對於 AD、MCI、HC 等縮寫，請以大寫英文字母保留。"
    "若有單字實在聽不清楚，請不要亂猜，可以以『[聽不清楚]』標示。"
)

LABEL_DIRS = {
    "AD": "AD",
    "HC": "HC",
    "MCI": "MCI",
}

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# ===== 產生 (label, sample_id, audio_path) =====
def iter_audio_files(data_root: Path):
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


def run_ncmmsc_asr(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    initial_prompt: str = DEFAULT_INITIAL_PROMPT,
):
    """
    NCMMSC 音檔 → Whisper ASR → 一個 CSV（含 transcript & cleaned_transcript）
    """
    data_root = Path(data_root)
    output_csv = Path(output_csv)

    print(f"[INFO] Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    fieldnames = ["id", "label", "transcript", "cleaned_transcript", "audio_path", "duration"]

    files = list(iter_audio_files(data_root))
    total_files = len(files)
    print(f"[INFO] Found {total_files} audio files under {data_root}")

    decode_kwargs = dict(
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

    ok_count = 0
    err_count = 0

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

    print(f"\n[INFO] Done. Saved {total_files} rows to {output_csv} (ok={ok_count}, error={err_count})")

# ===== CLI =====
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run Whisper ASR on NCMMSC audio and export CSV.")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="音檔根目錄")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="輸出 CSV 路徑")
    parser.add_argument("--model-size", type=str, default=DEFAULT_MODEL_SIZE, help="Whisper 模型大小")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="運算裝置：cuda / cpu")
    parser.add_argument("--compute-type", type=str, default=DEFAULT_COMPUTE_TYPE, help="計算型態：float16 等")
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
    )

if __name__ == "__main__":
    cli_main()
