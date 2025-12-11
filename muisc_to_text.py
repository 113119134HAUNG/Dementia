# muisc_to_text.py

# ========= 基本套件 =========
import os
import csv
import re
from pathlib import Path
from faster_whisper import WhisperModel

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ========= 預設設定（當成預設參數用） =========
DEFAULT_DATA_ROOT = Path("/content/NCMMSC2021_AD_Competition-dev/dataset/merge")
DEFAULT_OUTPUT_CSV = Path("ncmmsc_merged_asr_transcripts.csv")

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

# ---- 簡單中文清理，跟預處理 JSONL 版同精神 ----
ZHUYIN_PATTERN = re.compile(r"[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+")

def clean_chinese_transcript(text: str) -> str:
    t = str(text)
    t = t.replace("[聽不清楚]", " ")
    t = ZHUYIN_PATTERN.sub("嗯", t)
    t = re.sub(r"(嗯[\s、，,.!?]*){2,}", "嗯 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def iter_audio_files(data_root: Path, label_dirs: dict):
    """
    產生 (label, sample_id, audio_path)。
    data_root: 根資料夾，例如 /content/.../merge
    """
    for label, subdir in label_dirs.items():
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


# ========= 工具函式：給 notebook 或其他程式呼叫 =========
def run_transcription(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    initial_prompt: str = DEFAULT_INITIAL_PROMPT,
):
    """
    主要的轉錄流程：
    - data_root: 音檔的根目錄（底下有 AD/HC/MCI 子資料夾）
    - output_csv: 輸出的 CSV 檔名 / 路徑
    - model_size, device, compute_type, initial_prompt: Whisper 相關設定
    """

    print(f"Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    fieldnames = ["id", "label", "transcript", "cleaned_transcript",
                  "audio_path", "duration"]

    files = list(iter_audio_files(data_root, LABEL_DIRS))
    total_files = len(files)
    print(f"Found {total_files} audio files.")

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

    # 確保 output_csv 是 Path 物件
    output_csv = Path(output_csv)

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
                full_text = " ".join(
                    seg.text.strip() for seg in segments if seg.text.strip()
                )
                duration = getattr(info, "duration", 0.0)
                cleaned = clean_chinese_transcript(full_text)
                ok_count += 1
            except Exception as e:
                print(f"[ERROR] 無法轉錄 {audio_path}: {e}")
                full_text = ""
                cleaned = ""
                duration = 0.0
                err_count += 1

            writer.writerow({
                "id": sample_id,
                "label": label,
                "transcript": full_text,
                "cleaned_transcript": cleaned,
                "audio_path": audio_path,
                "duration": duration,
            })

    print(f"\nDone. Saved {total_files} rows to {output_csv} "
          f"(ok={ok_count}, error={err_count})")


# ========= 保留 CLI 支援 =========
def main():
    run_transcription()

if __name__ == "__main__":
    main()
