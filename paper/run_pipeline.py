# -*- coding: utf-8 -*-
"""
paper/run_pipeline.py  (paper-style, clean, argparse-managed)

One-command pipeline runner:
0) Music_to_text.asr_ncmmsc           (ASR -> CSV)
1) preprocess.preprocess_predictive   (TSV -> JSONL)
2) preprocess.* (text pipeline)       (merge/clean -> writes text.cleaned_jsonl)
3) paper.evaluate_cv                  (CV)

- No notebook magics
- No manual PYTHONPATH needed (script injects repo root)
- Text pipeline module auto-discovered, or explicitly set via --text-module
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path(repo: Path) -> None:
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)

def _run_module(module: str, *, config_path: Path, repo: Path, extra_args: Optional[List[str]] = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd += list(extra_args)

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def _find_text_pipeline_module(repo: Path) -> Optional[str]:
    """
    Auto-discover a text preprocessing entry under preprocess/.
    Your repo uses preprocess/preprocess_chinese.py, so include it explicitly.
    """
    preferred = [
        "preprocess.preprocess_chinese",
        "preprocess.preprocess_text",
        "preprocess.preprocess_text_merge",
        "preprocess.preprocess_text_build",
        "preprocess.preprocess_text_clean",
        "preprocess.preprocess_texts",
        "preprocess.preprocess_text_dataset",
    ]
    for m in preferred:
        if importlib.util.find_spec(m) is not None:
            return m

    pre_dir = repo / "preprocess"
    if not pre_dir.exists():
        return None

    keywords = ("text", "chinese", "clean", "merge")
    stems: List[str] = []
    for p in sorted(pre_dir.glob("*.py")):
        if p.name.startswith("_") or p.stem == "__init__":
            continue
        stem_l = p.stem.lower()
        if any(k in stem_l for k in keywords):
            stems.append(p.stem)

    for stem in stems:
        m = f"preprocess.{stem}"
        if importlib.util.find_spec(m) is not None:
            return m

    return None

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-style one-command pipeline runner (YAML-driven).")

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config_text.yaml (default: <repo>/settings/config_text.yaml)",
    )

    # steps
    p.add_argument("--asr", action="store_true", help="Run Music_to_text.asr_ncmmsc (ASR -> CSV).")
    p.add_argument("--predictive", action="store_true", help="Run preprocess_predictive (TSV -> JSONL).")
    p.add_argument("--text", action="store_true", help="Run text preprocessing pipeline (merge/clean).")
    p.add_argument("--cv", action="store_true", help="Run paper.evaluate_cv.")
    p.add_argument("--all", action="store_true", help="Run ASR + predictive + text + cv.")

    # ASR passthrough (optional)
    p.add_argument("--asr-labels", nargs="+", default=None, help="Pass to ASR: --labels AD HC (optional).")
    p.add_argument("--asr-cap-per-label", type=int, default=None, help="Pass to ASR: --cap-per-label N (optional).")

    # overrides
    p.add_argument(
        "--text-module",
        type=str,
        default=None,
        help="Explicit text pipeline module, e.g., preprocess.preprocess_chinese",
    )

    # CV knobs
    p.add_argument("--reuse-folds", action="store_true", help="Reuse existing folds_indices.json in config.")

    return p

def cli_main() -> None:
    repo = _repo_root()
    _ensure_repo_on_path(repo)

    args = build_arg_parser().parse_args()
    config_path = Path(args.config) if args.config else (repo / "settings" / "config_text.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    do_all = bool(args.all)
    do_asr = do_all or bool(args.asr)
    do_pred = do_all or bool(args.predictive)
    do_text = do_all or bool(args.text)
    do_cv = do_all or bool(args.cv)

    from tools.config_utils import load_text_config, get_text_config  # noqa: E402

    cfg = load_text_config(str(config_path))
    print("[INFO] TOP LEVEL KEYS:", list(cfg.keys()))

    # 0) ASR
    if do_asr:
        asr_extra: List[str] = []
        if args.asr_labels:
            asr_extra += ["--labels"] + list(args.asr_labels)
        if args.asr_cap_per_label is not None:
            asr_extra += ["--cap-per-label", str(int(args.asr_cap_per_label))]

        # Prefer in-process call if available; fallback to -m
        try:
            from Music_to_text.asr_ncmmsc import run_ncmmsc_asr  # type: ignore

            print("[INFO] Running Music_to_text.asr_ncmmsc (in-process)")
            run_ncmmsc_asr(
                config_path=str(config_path),
                labels=list(args.asr_labels) if args.asr_labels else None,
                cap_per_label=int(args.asr_cap_per_label) if args.asr_cap_per_label is not None else None,
            )
        except Exception as e:
            print("[WARN] In-process ASR failed; fallback to module run. Error:", repr(e))
            _run_module("Music_to_text.asr_ncmmsc", config_path=config_path, repo=repo, extra_args=asr_extra)

    # 1) predictive (TSV -> JSONL)
    if do_pred:
        try:
            from preprocess.preprocess_predictive import run_predictive_preprocessing  # type: ignore

            print("[INFO] Running preprocess_predictive (in-process)")
            run_predictive_preprocessing(config_path=str(config_path))
        except Exception as e:
            print("[WARN] In-process predictive failed; fallback to module run. Error:", repr(e))
            _run_module("preprocess.preprocess_predictive", config_path=config_path, repo=repo)

    # 2) text pipeline (merge/clean)
    if do_text:
        mod = (args.text_module or "").strip() or _find_text_pipeline_module(repo)
        if not mod:
            pre_dir = repo / "preprocess"
            files = [p.name for p in sorted(pre_dir.glob("*.py"))] if pre_dir.exists() else []
            raise RuntimeError(
                "Cannot find a text preprocessing entry module under preprocess/. "
                f"Found: {files}\n"
                "Fix: run with --text-module preprocess.preprocess_chinese"
            )

        print(f"[INFO] Running text pipeline module: {mod}")
        _run_module(mod, config_path=config_path, repo=repo)

        # verify cleaned_jsonl exists
        text_cfg = get_text_config(cfg=cfg)
        cleaned = Path(str(text_cfg.get("cleaned_jsonl", "")))
        if not cleaned.exists():
            raise FileNotFoundError(
                f"cleaned_jsonl not found after text pipeline: {cleaned}\n"
                f"Check your text preprocessing script: {mod}"
            )
        print("[OK] cleaned_jsonl:", cleaned)

    # 3) CV
    if do_cv:
        try:
            from paper.evaluate_cv import run_evaluate_cv  # type: ignore  # noqa: E402

            print("[INFO] Running paper.evaluate_cv (in-process)")
            run_evaluate_cv(
                config_path=str(config_path),
                methods_override=None,
                reuse_folds=bool(args.reuse_folds),
            )
        except Exception as e:
            print("[WARN] In-process CV failed; fallback to module run. Error:", repr(e))
            # paper.evaluate_cv should also accept --config
            extra = ["--reuse-folds"] if bool(args.reuse_folds) else None
            _run_module("paper.evaluate_cv", config_path=config_path, repo=repo, extra_args=extra)

    print("\n===== DONE =====")
    print("config:", config_path)

if __name__ == "__main__":
    cli_main()
