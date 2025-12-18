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
- Colab/Jupyter safe (ignores injected argv like -f ...)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _repo_root() -> Path:
    # file: <repo>/paper/run_pipeline.py -> parents[1] == <repo>
    return Path(__file__).resolve().parents[1]


def _run_module(
    module: str,
    *,
    config_path: Path,
    repo: Path,
    extra_args: Optional[List[str]] = None,
    step_name: Optional[str] = None,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    if extra_args:
        cmd.extend([str(x) for x in extra_args if str(x).strip()])

    tag = step_name or module
    print(f"\n===== RUN STEP: {tag} =====")
    print("$ " + " ".join(cmd))

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


def _parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, unknown


def cli_main(args: argparse.Namespace, *, unknown_args: Optional[List[str]] = None) -> None:
    repo = _repo_root()

    config_path = Path(args.config) if args.config else (repo / "settings" / "config_text.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # sanity check YAML loads (in current process)
    sys.path.insert(0, str(repo))
    from tools.config_utils import load_text_config  # noqa: E402

    cfg = load_text_config(str(config_path))
    print("[INFO] repo:", repo)
    print("[INFO] config:", config_path)
    print("[INFO] TOP LEVEL KEYS:", list(cfg.keys()))

    if unknown_args:
        # Colab/Jupyter often injects args like: -f <kernel.json>
        print("[INFO] Ignored unknown args:", unknown_args)

    if not (args.asr or args.predictive or args.text or args.cv):
        raise SystemExit("No step specified. Use one or more: --asr --predictive --text --cv")

    # run steps in fixed order (reproducible)
    if args.asr:
        _run_module(
            "Music_to_text.asr_ncmmsc",
            config_path=config_path,
            repo=repo,
            step_name="asr",
        )

    if args.predictive:
        _run_module(
            "preprocess.preprocess_predictive",
            config_path=config_path,
            repo=repo,
            step_name="predictive",
        )

    if args.text:
        extra: List[str] = []
        if bool(args.force_asr_jsonl):
            extra.append("--force-asr-jsonl")
        if bool(args.skip_asr):
            extra.append("--skip-asr")

        _run_module(
            "preprocess.preprocess_chinese",
            config_path=config_path,
            repo=repo,
            extra_args=extra,
            step_name="text",
        )

    if args.cv:
        extra2: List[str] = []
        if bool(args.reuse_folds):
            extra2.append("--reuse-folds")

        _run_module(
            "paper.evaluate_cv",
            config_path=config_path,
            repo=repo,
            extra_args=extra2,
            step_name="cv",
        )

    print("\n===== DONE =====")
    print("config:", config_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, unknown = _parse_args(argv)
    cli_main(args, unknown_args=unknown)


if __name__ == "__main__":
    main()
