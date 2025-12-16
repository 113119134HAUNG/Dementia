# -*- coding: utf-8 -*-
"""
asr_ncmmsc.py (paper-strict, argparse-managed)

- YAML is the single source of truth (load once).
- Deterministic file discovery order.
- Writes RAW transcript only (NO cleaning).
  All cleaning happens later in preprocess/preprocess_chinese.py (single point).
- Output CSV columns are FIXED:
    id,label,transcript,audio_path,duration

Fail-fast:
- if no audio files found, raise FileNotFoundError
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from faster_whisper import WhisperModel

from tools.config_utils import load_text_config, get_asr_config

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

LABEL_DIRS: Dict[str, str] = {"AD": "AD", "HC": "HC", "MCI": "MCI"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

def _normalize_label(x: str) -> str:
    return str(x).strip().upper()

def _make_sample_id(label: str, stem: str) -> str:
    """Avoid AD_AD_0001 when stem already has label prefix."""
    lb = _normalize_label(label)
    st = str(stem).strip()
    if st.upper().startswith(f"{lb}_"):
        return st
    return f"{lb}_{st}"

def iter_audio_files(
    data_root: Path,
    *,
    labels: Optional[List[str]] = None,
    cap_per_label: Optional[int] = None,
) -> Iterable[Tuple[str, str, str]]:
    allow = None if labels is None else {_normalize_label(x) for x in labels}
    counts: Dict[str, int] = {}

    for label in ("AD", "HC", "MCI"):
        if allow is not None and label not in allow:
            continue

        audio_dir = data_root / LABEL_DIRS[label]
        if not audio_dir.is_dir():
            print(f"[WARN] Missing directory, skip: {audio_dir}")
            continue

        counts.setdefault(label, 0)

        for path in sorted(audio_dir.iterdir(), key=lambda p: p.name):  # deterministic
            if cap_per_label is not None and counts[label] >= int(cap_per_label):
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in AUDIO_EXTS:
                continue

            sample_id = _make_sample_id(label, path.stem)
            counts[label] += 1
            yield label, sample_id, str(path)

def _build_decode_kwargs(asr_cfg: Dict[str, Any]) -> Dict[str, Any]:
    decode_cfg = asr_cfg.get("decode")
    if decode_cfg is None or not isinstance(decode_cfg, dict):
        raise KeyError("Missing asr.decode (dict) in YAML.")
    return dict(decode_cfg)

def run_ncmmsc_asr(
    config_path: Optional[str] = None,
    *,
    labels: Optional[List[str]] = None,
    cap_per_label: Optional[int] = None,
) -> None:
    cfg = load_text_config(config_path)
    asr_cfg = get_asr_config(cfg=cfg)

    data_root = Path(asr_cfg["data_root"])
    output_csv = Path(asr_cfg["output_csv"])
    model_size = str(asr_cfg["model_size"])
    device = str(asr_cfg["device"])
    compute_type = str(asr_cfg["compute_type"])
    initial_prompt = str(asr_cfg.get("initial_prompt", "") or "")

    if not data_root.exists():
        raise FileNotFoundError(f"ASR data_root not found: {data_root}")

    files = list(iter_audio_files(data_root, labels=labels, cap_per_label=cap_per_label))
    print(f"[INFO] Found {len(files)} audio files under {data_root}")
    if len(files) == 0:
        raise FileNotFoundError(f"No audio files found under: {data_root}")

    print(f"[INFO] Loading Whisper model: {model_size} on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    decode_kwargs = _build_decode_kwargs(asr_cfg)
    if initial_prompt:
        decode_kwargs["initial_prompt"] = initial_prompt

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    err_count = 0

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "label", "transcript", "audio_path", "duration"],
        )
        w.writeheader()

        iterable = tqdm(files, desc="Transcribing", unit="file", leave=True) if HAS_TQDM else files

        for label, sample_id, audio_path in iterable:
            if HAS_TQDM:
                iterable.set_postfix_str(Path(audio_path).name)
            else:
                print(f"[INFO] Transcribing {audio_path} ...")

            try:
                segments, info = model.transcribe(audio_path, **decode_kwargs)
                raw_text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())
                duration = float(getattr(info, "duration", 0.0) or 0.0)
                ok_count += 1
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Failed to transcribe {audio_path}: {e}")
                raw_text = ""
                duration = 0.0
                err_count += 1

            w.writerow(
                {
                    "id": sample_id,
                    "label": _normalize_label(label),
                    "transcript": raw_text,
                    "audio_path": audio_path,
                    "duration": duration,
                }
            )

    print(f"\n[INFO] Done. Saved {len(files)} rows to {output_csv} (ok={ok_count}, error={err_count})")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Whisper ASR on NCMMSC audio and export raw CSV (paper-strict; single-point cleaning later)."
    )
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument("--labels", nargs="+", default=None, help="Optional subset labels. Example: --labels AD HC")
    p.add_argument("--cap-per-label", type=int, default=None, help="Optional cap per label. Example: --cap-per-label 200")
    return p

def cli_main() -> None:
    args = build_arg_parser().parse_args()
    run_ncmmsc_asr(config_path=args.config, labels=args.labels, cap_per_label=args.cap_per_label)

if __name__ == "__main__":
    cli_main()
