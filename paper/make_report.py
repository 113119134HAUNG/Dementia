# -*- coding: utf-8 -*-
"""
make_report.py

Generate paper-style artifacts from existing outputs:
1) Data distribution (2D projection + optional logistic decision boundary)
2) Results summary table (mean ± std) from cv_metrics.json
3) Boxplot of accuracy across folds (+ mean marker + median line + p-value)

Works in Colab/Jupyter (ignores injected argv like -f).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

from tools.config_utils import load_text_config, get_text_config, get_features_config
from paper.cv_utils import require, get_dict, ensure_parent
from paper.cv_dataset import load_cleaned_dataset
from paper.cv_features import tfidf_features_all

# -----------------------------
# helpers
# -----------------------------
def _norm_method_name(m: str) -> str:
    m = (m or "").strip().lower()
    if m == "bert":
        return "BERT-base (average)"
    if m == "tfidf":
        return "Tf-Idf"
    if m == "glove":
        return "GloVe (300 d)"
    if m == "gemma":
        return "Gemma-2B"
    return m

def _fmt_mean_std(mean: float, std: float, *, scale_100: bool = True, digits: int = 2) -> str:
    if scale_100:
        mean *= 100.0
        std *= 100.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _safe_load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# -----------------------------
# 1) distribution plot (2D + optional boundary)
# -----------------------------
def plot_distribution_2d_tfidf(
    *,
    X_text: List[str],
    y: np.ndarray,
    label_names: List[str],
    tfidf_cfg: Dict[str, Any],
    out_png: Path,
    title: str = "Logistic Regression Decision Boundary (2D projection)",
    draw_boundary: bool = True,
    random_state: int = 42,
) -> None:
    vect_cfg = get_dict(tfidf_cfg, "vectorizer", where="features.tfidf")
    trf_cfg = get_dict(tfidf_cfg, "transformer", where="features.tfidf")

    # High-dim sparse -> 2D (SVD is PCA-like for sparse)
    X_all = tfidf_features_all(X_text, vectorizer_cfg=vect_cfg, transformer_cfg=trf_cfg)
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    X2 = svd.fit_transform(X_all)

    plt.figure()
    ax = plt.gca()

    if draw_boundary:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        clf.fit(X2, y)

        x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
        y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 250),
            np.linspace(y_min, y_max, 250),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        prob = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        # background + boundary at 0.5
        plt.contourf(xx, yy, prob, alpha=0.15)
        plt.contour(xx, yy, prob, levels=[0.5], linestyles="--")

    # Scatter by class (use default color cycle; do not set colors explicitly)
    for k in sorted(np.unique(y)):
        mask = (y == k)
        name = label_names[int(k)] if int(k) < len(label_names) else f"class_{k}"
        plt.scatter(X2[mask, 0], X2[mask, 1], label=name, s=35)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()

    ensure_parent(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------------
# 2) results table from cv_metrics.json
# -----------------------------
def build_results_table(
    *,
    cv_metrics: Dict[str, Any],
    out_csv: Path,
    out_tex: Path,
) -> pd.DataFrame:
    methods = cv_metrics.get("methods", {}) or {}

    rows = []
    for m_key, m_val in methods.items():
        sm = (m_val or {}).get("summary", {}) or {}
        rows.append(
            {
                "Embedding Model": _norm_method_name(m_key),
                "Accuracy (%)": _fmt_mean_std(sm.get("accuracy_mean", np.nan), sm.get("accuracy_std", np.nan)),
                "Precision (%)": _fmt_mean_std(sm.get("precision_mean", np.nan), sm.get("precision_std", np.nan)),
                "Recall (%)": _fmt_mean_std(sm.get("recall_mean", np.nan), sm.get("recall_std", np.nan)),
                "F1-Score (%)": _fmt_mean_std(sm.get("f1_mean", np.nan), sm.get("f1_std", np.nan)),
            }
        )

    df = pd.DataFrame(rows)

    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # LaTeX table (simple, paper-ready)
    ensure_parent(out_tex)
    out_tex.write_text(df.to_latex(index=False), encoding="utf-8")

    return df

# -----------------------------
# 3) boxplot accuracy across folds (+ p-value)
# -----------------------------
def plot_accuracy_boxplot(
    *,
    cv_metrics: Dict[str, Any],
    out_png: Path,
    title: str = "Accuracy across Cross-Validation Folds",
) -> None:
    methods = cv_metrics.get("methods", {}) or {}

    labels = []
    data = []

    for m_key, m_val in methods.items():
        fold_metrics = (m_val or {}).get("fold_metrics", []) or []
        accs = [float(r.get("accuracy", np.nan)) for r in fold_metrics if "accuracy" in r]
        if len(accs) == 0:
            continue
        labels.append(_norm_method_name(m_key))
        data.append(np.array(accs, dtype=float) * 100.0)  # percent

    plt.figure()
    ax = plt.gca()

    bp = plt.boxplot(data, labels=labels, showfliers=True)

    # mean markers
    means = [float(np.nanmean(x)) for x in data]
    xs = np.arange(1, len(means) + 1)
    plt.scatter(xs, means, marker="^", label="Mean")

    # p-value (Friedman, repeated measures across folds)
    p_val_txt = ""
    try:
        from scipy.stats import friedmanchisquare
        if len(data) >= 2:
            # requires same length per method; if not, truncate to min
            min_len = min(len(x) for x in data)
            arrs = [x[:min_len] for x in data]
            stat, p = friedmanchisquare(*arrs)
            p_val_txt = f"p_val = {p:.2f}"
    except Exception:
        p_val_txt = ""

    plt.title(title)
    plt.ylabel("Accuracy (%)")

    # legend text (mean + p_val)
    if p_val_txt:
        plt.legend(title=p_val_txt)
    else:
        plt.legend()

    plt.tight_layout()
    ensure_parent(out_png)
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------------
# main
# -----------------------------
def run_make_report(
    *,
    config_path: Optional[str],
    outdir: str,
    dist_method: str = "tfidf",
    draw_boundary: bool = True,
) -> None:
    cfg = load_text_config(config_path)
    text_cfg = get_text_config(cfg=cfg)
    feat_cfg = get_features_config(cfg=cfg)

    cleaned_jsonl = Path(str(require(text_cfg, "cleaned_jsonl", where="text")))
    cv_cfg = get_dict(feat_cfg, "crossval", where="features")
    metrics_output = Path(str(require(cv_cfg, "metrics_output", where="features.crossval")))

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load cleaned dataset
    data = load_cleaned_dataset(cleaned_jsonl)
    X_text = data.X
    y = data.y
    label_names = data.label_names

    # 1) distribution
    if dist_method.strip().lower() == "tfidf":
        tfidf_cfg = get_dict(feat_cfg, "tfidf", where="features")
        plot_distribution_2d_tfidf(
            X_text=X_text,
            y=y,
            label_names=label_names,
            tfidf_cfg=tfidf_cfg,
            out_png=out_dir / "dist_2d_tfidf.png",
            draw_boundary=bool(draw_boundary),
            random_state=int(get_dict(cfg, "common", where="root").get("seed", 42)),
        )

    # 2/3) metrics-based artifacts
    if metrics_output.exists():
        cv_metrics = _safe_load_json(metrics_output)

        build_results_table(
            cv_metrics=cv_metrics,
            out_csv=out_dir / "results_table.csv",
            out_tex=out_dir / "results_table.tex",
        )

        plot_accuracy_boxplot(
            cv_metrics=cv_metrics,
            out_png=out_dir / "accuracy_boxplot.png",
        )
    else:
        print(f"[WARN] metrics_output not found: {metrics_output} (skip table/boxplot)")

    print(f"[INFO] Report outputs saved to: {out_dir}")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate distribution plot + results table + boxplot from pipeline outputs.")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument("--outdir", type=str, default="/content/chinese_combined/report", help="Output directory")
    p.add_argument("--dist-method", type=str, default="tfidf", help="Distribution method: tfidf (more later if needed)")
    p.add_argument("--no-boundary", action="store_true", help="Disable decision boundary on distribution plot.")
    return p

def cli_main() -> None:
    args, _unknown = build_arg_parser().parse_known_args()  # ignore Colab/Jupyter injected args (-f ...)
    run_make_report(
        config_path=args.config,
        outdir=args.outdir,
        dist_method=args.dist_method,
        draw_boundary=not bool(args.no_boundary),
    )

if __name__ == "__main__":
    cli_main()
