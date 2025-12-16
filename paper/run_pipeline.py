# -*- coding: utf-8 -*-
"""
paper/run_pipeline.py  (paper-style, clean, argparse-managed)

One-command pipeline runner (after setup_colab finished):
1) preprocess.preprocess_predictive
2) preprocess.*(text pipeline) -> writes text.cleaned_jsonl
3) paper.evaluate_cv

- No notebook magics
- No manual PYTHONPATH needed (script injects repo root)
- Text pipeline module is auto-discovered under preprocess/ (text-related scripts)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

def _repo_root() -> Path:
    # /content/Dementia/paper/run_pipeline.py -> repo root is parent of "paper"
    return Path(__file__).resolve().parents[1]

def _ensure_repo_on_path(repo: Path) -> None:
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)

def _run_module(module: str, *, config_path: Path, repo: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", module, "--config", str(config_path)]
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def _find_text_pipeline_module(repo: Path) -> Optional[str]:
    """
    Auto-discover a text preprocessing entry under preprocess/.
    Strategy:
      1) try known canonical names (most likely)
      2) otherwise scan preprocess/*.py containing 'text' in filename
    """
    # 1) preferred names (keep your project naming here)
    preferred = [
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

    # 2) scan directory for anything with "text" in filename
    pre_dir = repo / "preprocess"
    if not pre_dir.exists():
        return None

    stems = []
    for p in sorted(pre_dir.glob("*.py")):
        if p.name.startswith("_"):
            continue
        if p.stem == "__init__":
            continue
        if "text" in p.stem.lower():
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
    p.add_argument("--predictive", action="store_true", help="Run preprocess_predictive.")
    p.add_argument("--text", action="store_true", help="Run text preprocessing pipeline (merge/clean).")
    p.add_argument("--cv", action="store_true", help="Run paper.evaluate_cv.")
    p.add_argument("--all", action="store_true", help="Run predictive + text + cv.")

    # CV knobs passthrough
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
    do_pred = do_all or bool(args.predictive)
    do_text = do_all or bool(args.text)
    do_cv = do_all or bool(args.cv)

    # quick sanity check for top-level YAML keys (early fail)
    from tools.config_utils import load_text_config, get_text_config  # noqa

    cfg = load_text_config(str(config_path))
    print("[INFO] TOP LEVEL KEYS:", list(cfg.keys()))

    # predictive
    if do_pred:
        # Prefer in-process call if available; fallback to -m
        try:
            from preprocess.preprocess_predictive import run_predictive_preprocessing  # type: ignore

            print("[INFO] Running preprocess_predictive (in-process)")
            run_predictive_preprocessing(config_path=str(config_path))
        except Exception as e:
            print("[WARN] In-process predictive failed; fallback to module run. Error:", repr(e))
            _run_module("preprocess.preprocess_predictive", config_path=config_path, repo=repo)

    # text pipeline (must produce cleaned_jsonl)
    if do_text:
        mod = _find_text_pipeline_module(repo)
        if not mod:
            pre_dir = repo / "preprocess"
            files = [p.name for p in sorted(pre_dir.glob("*.py"))] if pre_dir.exists() else []
            raise RuntimeError(
                "Cannot find a text preprocessing entry module under preprocess/. "
                f"Found: {files}"
            )
        print(f"[INFO] Running text pipeline module: {mod}")
        _run_module(mod, config_path=config_path, repo=repo)

        # verify cleaned_jsonl exists (from YAML)
        text_cfg = get_text_config(cfg=cfg)
        cleaned = Path(str(text_cfg.get("cleaned_jsonl", "")))
        if cleaned and cleaned.exists():
            print("[OK] cleaned_jsonl:", cleaned)
        else:
            print("[WARN] cleaned_jsonl not found yet (check your text pipeline step):", cleaned)

    # 3) CV
    if do_cv:
        from paper.evaluate_cv import run_evaluate_cv  # type: ignore

        print("[INFO] Running paper.evaluate_cv")
        run_evaluate_cv(
            config_path=str(config_path),
            methods_override=None,
            reuse_folds=bool(args.reuse_folds),
        )

    print("\n===== DONE =====")
    print("config:", config_path)

if __name__ == "__main__":
    cli_main()
