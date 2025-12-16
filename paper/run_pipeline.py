# -*- coding: utf-8 -*-
"""
paper/run_pipeline.py  (paper-style, argparse-managed)

Multi-step runner (NO --all):
- asr:        Music_to_text.asr_ncmmsc
- predictive: preprocess.preprocess_predictive
- text:       preprocess.preprocess_chinese
- cv:         paper.evaluate_cv

Rationale:
- explicit, debuggable, reproducible
- no implicit "all-in-one" side effects
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _run_module(module: str, *, config_path: Path, repo: Path, extra_args: Optional[List[str]] = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-style pipeline runner (STEP-BY-STEP; no --all).")

    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config_text.yaml (default: <repo>/settings/config_text.yaml)",
    )

    # steps (explicit)
    p.add_argument("--asr", action="store_true", help="Run Music_to_text.asr_ncmmsc")
    p.add_argument("--predictive", action="store_true", help="Run preprocess.preprocess_predictive")
    p.add_argument("--text", action="store_true", help="Run preprocess.preprocess_chinese")
    p.add_argument("--cv", action="store_true", help="Run paper.evaluate_cv")

    # passthrough flags
    p.add_argument("--force-asr-jsonl", action="store_true", help="Pass to preprocess.preprocess_chinese")
    p.add_argument("--skip-asr", action="store_true", help="Pass to preprocess.preprocess_chinese (skip ASR->JSONL step)")
    p.add_argument("--reuse-folds", action="store_true", help="Pass to paper.evaluate_cv")

    return p

def cli_main() -> None:
    repo = _repo_root()
    config_path = Path(args.config) if args.config else (repo / "settings" / "config_text.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # sanity check YAML loads
    sys.path.insert(0, str(repo))
    from tools.config_utils import load_text_config  # noqa: E402

    cfg = load_text_config(str(config_path))
    print("[INFO] TOP LEVEL KEYS:", list(cfg.keys()))

    if not (args.asr or args.predictive or args.text or args.cv):
        raise SystemExit("No step specified. Use one or more: --asr --predictive --text --cv")

    if args.asr:
        _run_module("Music_to_text.asr_ncmmsc", config_path=config_path, repo=repo)

    if args.predictive:
        _run_module("preprocess.preprocess_predictive", config_path=config_path, repo=repo)

    if args.text:
        extra: List[str] = []
        if args.force_asr_jsonl:
            extra.append("--force-asr-jsonl")
        if args.skip_asr:
            extra.append("--skip-asr")
        _run_module("preprocess.preprocess_chinese", config_path=config_path, repo=repo, extra_args=extra)

    if args.cv:
        extra2: List[str] = []
        if args.reuse_folds:
            extra2.append("--reuse-folds")
        _run_module("paper.evaluate_cv", config_path=config_path, repo=repo, extra_args=extra2)

    print("\n===== DONE =====")
    print("config:", config_path)

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    cli_main()
