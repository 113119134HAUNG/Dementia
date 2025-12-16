# -*- coding: utf-8 -*-
"""
setup_colab.py

One-shot Colab setup script:
- optional pip install (Colab-safe by default; avoids upgrading core libs)
- download + unzip dataset zip (Google Drive)
- clone repo
- download fastText Chinese vectors (.vec)
- Hugging Face login + pre-download models

No notebook magics. No scattered shell commands.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

# -----------------------------
# Shell helper
# -----------------------------
def run(cmd: List[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def in_colab() -> bool:
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ

# -----------------------------
# Install deps (Colab-safe)
# -----------------------------
def install_deps(
    *,
    quiet: bool = True,
    upgrade: bool = False,
    install_core: bool = False,
    install_torch: bool = False,
) -> None:
    """
    Colab-safe install:
    - default: DO NOT touch numpy/pandas/torch (avoid conflicts with colab/opencv/tf/torchvision)
    - optional flags to include them if you really need.
    """
    # packages that are usually safe to install on Colab without breaking preinstalls
    pkgs = [
        "scikit-learn",
        "jieba",
        "transformers",
        "accelerate",
        "sentencepiece",
        "huggingface_hub",
        "hf_transfer",
        "faster-whisper",
        "gdown",
    ]

    # Only install/upgrade these when explicitly asked.
    # On Colab, upgrading them often causes conflicts with preinstalled opencv/tf/torchvision/torchaudio.
    if install_core:
        pkgs = ["numpy", "pandas"] + pkgs
    if install_torch:
        pkgs = ["torch"] + pkgs

    cmd: List[str] = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    if upgrade:
        cmd += ["-U", "--upgrade-strategy", "only-if-needed"]
    cmd += pkgs

    if in_colab() and (upgrade or install_core or install_torch):
        print(
            "[WARN] You enabled upgrades/core installs. On Colab this can cause dependency conflicts.\n"
            "       If you see numpy/pandas/torch conflicts, consider restarting runtime and running without these flags."
        )

    run(cmd, quiet=quiet)

# -----------------------------
# Google Drive download (gdown)
# -----------------------------
def gdrive_download(*, file_id: str, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    try:
        import gdown  # type: ignore
    except Exception as e:
        raise ImportError("gdown is required. Run with --install first.") from e

    url = f"https://drive.google.com/uc?id={file_id}"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[INFO] Zip exists, skip download: {output_path}")
        return output_path

    print(f"[INFO] Downloading from Google Drive id={file_id} -> {output_path}")
    out = gdown.download(url, str(output_path), quiet=False)
    if out is None:
        raise RuntimeError("gdown download failed (returned None).")
    return output_path

# -----------------------------
# Unzip
# -----------------------------
def unzip_file(zip_path: Path, *, out_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    ensure_dir(out_dir)
    print(f"[INFO] Unzipping: {zip_path} -> {out_dir}")

    unzip_bin = shutil.which("unzip")
    if unzip_bin:
        run([unzip_bin, "-o", "-q", str(zip_path), "-d", str(out_dir)], quiet=True)
        return

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=str(out_dir))

# -----------------------------
# Git clone
# -----------------------------
def git_clone(*, repo_url: str, repo_dir: Path) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"[INFO] Repo exists, skip clone: {repo_dir}")
        return
    ensure_dir(repo_dir.parent)
    print(f"[INFO] Cloning: {repo_url} -> {repo_dir}")
    run(["git", "clone", repo_url, str(repo_dir)], quiet=False)

# -----------------------------
# Download + gunzip fastText vectors
# -----------------------------
def download_fasttext_vec(*, url: str, out_vec_path: Path) -> Path:
    ensure_dir(out_vec_path.parent)
    if out_vec_path.exists() and out_vec_path.stat().st_size > 0:
        print(f"[INFO] fastText vec exists, skip: {out_vec_path}")
        return out_vec_path

    gz_path = out_vec_path.with_suffix(out_vec_path.suffix + ".gz")
    print(f"[INFO] Downloading fastText: {url} -> {gz_path}")

    with urllib.request.urlopen(url) as resp, open(gz_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"[INFO] Decompressing: {gz_path} -> {out_vec_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_vec_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    try:
        gz_path.unlink(missing_ok=True)
    except Exception:
        pass

    print(
        f"[OK] fastText ready: {out_vec_path}  "
        f"(size={out_vec_path.stat().st_size/1024/1024:.1f} MB)"
    )
    return out_vec_path

# -----------------------------
# HF login + model download (robust)
# -----------------------------
def hf_login_and_download(
    *,
    models: List[str],
    models_dir: Path,
    hf_token: Optional[str],
    interactive: bool,
) -> List[Path]:
    os.environ.setdefault("HF_HOME", "/content/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/content/hf/transformers")
    os.environ.setdefault("HF_HUB_CACHE", "/content/hf/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/content/hf/datasets")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    try:
        from huggingface_hub import HfApi, login, snapshot_download  # type: ignore
    except Exception as e:
        raise ImportError("huggingface_hub is required. Run with --install first.") from e

    def try_whoami(tok: str) -> Optional[str]:
        tok = (tok or "").strip()
        if not tok:
            return None
        try:
            who = HfApi(token=tok).whoami()
            return who.get("name") or who.get("email") or "unknown"
        except Exception:
            return None

    def get_colab_secret_token(secret_key: str = "FACE") -> str:
        try:
            from google.colab import userdata  # type: ignore
            return (userdata.get(secret_key) or "").strip()
        except Exception:
            return ""

    token_arg = (hf_token or "").strip()
    token_secret = get_colab_secret_token("FACE")
    token_env = (os.environ.get("HF_TOKEN", "") or "").strip()

    token = token_arg or token_secret or token_env
    who = try_whoami(token) if token else None

    # 如果選到的 token 無效，而且 Secrets 有值，就用 Secrets 再試一次（避免被舊 env HF_TOKEN 干擾）
    if not who and token_secret:
        token = token_secret
        who = try_whoami(token)

    # still invalid -> prompt only if interactive
    if not who and interactive:
        from getpass import getpass
        token = getpass("Paste your Hugging Face token (hidden): ").strip()
        who = try_whoami(token) if token else None

    if token and who:
        os.environ["HF_TOKEN"] = token  # 強制本進程用這個（覆蓋可能的錯 env）
        login(token=token)
        print("[OK] HF login:", who)
    else:
        print(
            "[WARN] No valid HF token.\n"
            "       Check Colab Secrets (🔑) key 'FACE' exists and is a valid hf_... token.\n"
            "       If you want NO prompt, run with --no-interactive."
        )
        token = ""

    ensure_dir(models_dir)
    local_paths: List[Path] = []

    for repo_id in models:
        repo_id = str(repo_id).strip()
        if not repo_id:
            continue

        subdir = repo_id.replace("/", "__")
        local_dir = models_dir / subdir
        ensure_dir(local_dir)

        try:
            if any(local_dir.iterdir()):
                print(f"[INFO] Model exists, skip: {repo_id} -> {local_dir}")
                local_paths.append(local_dir)
                continue
        except Exception:
            pass

        print(f"[INFO] Downloading model snapshot: {repo_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=(token or None),
            )
            local_paths.append(local_dir)
            print(f"[OK] {repo_id} -> {local_dir}")
        except Exception as e:
            print(f"[WARN] Failed: {repo_id}  (common: gated / license not accepted / no permission)")
            print(f"       Error: {repr(e)}")

    return local_paths

# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab one-shot setup (paper-style, argparse-managed).")

    # toggles
    p.add_argument("--install", action="store_true", help="Install dependencies via pip (Colab-safe defaults).")
    p.add_argument("--download-data", action="store_true", help="Download dataset zip from Google Drive.")
    p.add_argument("--unzip-data", action="store_true", help="Unzip dataset zip.")
    p.add_argument("--clone-repo", action="store_true", help="Clone repo (no-op if already exists).")
    p.add_argument("--download-vec", action="store_true", help="Download + gunzip fastText cc.zh.300.vec.")
    p.add_argument("--hf", action="store_true", help="Hugging Face login + pre-download models.")
    p.add_argument("--all", action="store_true", help="Run all steps above.")

    # install behavior
    p.add_argument("--upgrade", action="store_true", help="pip install -U (NOT recommended on Colab unless needed).")
    p.add_argument(
        "--install-core",
        action="store_true",
        help="Also install/upgrade numpy+pandas (may cause Colab conflicts).",
    )
    p.add_argument(
        "--install-torch",
        action="store_true",
        help="Also install/upgrade torch (may break torchvision/torchaudio on Colab).",
    )

    # params
    p.add_argument("--gdrive-file-id", type=str, default="1NPE7aLlSqlKdOJE4l5HTN06DxKP73a-O")
    p.add_argument("--zip-path", type=str, default="/content/NCMMSC2021_AD_Competition-dev.zip")
    p.add_argument("--unzip-dir", type=str, default="/content")

    p.add_argument("--repo-url", type=str, default="https://github.com/113119134HAUNG/Dementia.git")
    p.add_argument("--repo-dir", type=str, default="/content/Dementia")

    p.add_argument(
        "--fasttext-url",
        type=str,
        default="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz",
    )
    p.add_argument("--vec-path", type=str, default="/content/embeddings/cc.zh.300.vec")

    p.add_argument("--models-dir", type=str, default="/content/models")
    p.add_argument(
        "--models",
        nargs="+",
        default=["bert-base-chinese", "google/gemma-2b"],
        help="HF repo ids to download. Example: --models bert-base-chinese google/gemma-2b",
    )

    p.add_argument("--hf-token", type=str, default=None, help="HF token (prefer env HF_TOKEN or Secrets).")
    p.add_argument("--no-interactive", action="store_true", help="Do not prompt for HF token; use env/arg only.")
    return p

def cli_main() -> None:
    args = build_arg_parser().parse_args()

    do_all = bool(args.all)
    do_install = do_all or bool(args.install)
    do_dl = do_all or bool(args.download_data)
    do_unzip = do_all or bool(args.unzip_data)
    do_clone = do_all or bool(args.clone_repo)
    do_vec = do_all or bool(args.download_vec)
    do_hf = do_all or bool(args.hf)

    zip_path = Path(args.zip_path)
    unzip_dir = Path(args.unzip_dir)

    if do_install:
        install_deps(
            quiet=True,
            upgrade=bool(args.upgrade),
            install_core=bool(args.install_core),
            install_torch=bool(args.install_torch),
        )

    if do_dl:
        gdrive_download(file_id=str(args.gdrive_file_id).strip(), output_path=zip_path)

    if do_unzip:
        unzip_file(zip_path, out_dir=unzip_dir)

    if do_clone:
        git_clone(repo_url=str(args.repo_url).strip(), repo_dir=Path(args.repo_dir))

    if do_vec:
        download_fasttext_vec(url=str(args.fasttext_url).strip(), out_vec_path=Path(args.vec_path))

    if do_hf:
        hf_login_and_download(
            models=[str(m).strip() for m in args.models if str(m).strip()],
            models_dir=Path(args.models_dir),
            hf_token=args.hf_token,
            interactive=not bool(args.no_interactive),
        )

    print("\n===== SUMMARY =====")
    print("zip_path   :", zip_path)
    print("unzip_dir  :", unzip_dir)
    print("repo_dir   :", args.repo_dir)
    print("vec_path   :", args.vec_path)
    print("models_dir :", args.models_dir)
    print("models     :", args.models)

if __name__ == "__main__":
    cli_main()
