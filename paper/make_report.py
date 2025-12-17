# -*- coding: utf-8 -*-
"""
paper/make_report.py

Generate paper-style artifacts from existing outputs (paper-strict):
Inputs:
- cleaned.jsonl (from preprocess.preprocess_chinese)
- cv_metrics.json (from paper.evaluate_cv)

Outputs (default outdir=/content/chinese_combined/report):
1) dist_2d_tfidf.png
2) results_table.csv + results_table.tex
3) boxplot_accuracy.png / boxplot_precision.png / boxplot_recall.png / boxplot_f1.png

Works in Colab/Jupyter (ignores injected argv like -f).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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
# naming / formatting helpers
# -----------------------------
METHOD_ORDER = ["bert", "tfidf", "gemma", "glove"]

DISPLAY_NAME = {
    "bert": "BERT-base (average)",
    "tfidf": "Tf-Idf",
    "glove": "GloVe (300 d)",
    "gemma": "Gemma-2B",
}

METRIC_INFO = {
    "accuracy": ("Accuracy (成功率/正確率) (%)", "Accuracy (%)"),
    "precision": ("Precision (精確率/準確率) (%)", "Precision (%)"),
    "recall": ("Recall (召回率) (%)", "Recall (%)"),
    "f1": ("F1-Score (%)", "F1-Score (%)"),
}

def _norm_method_key(m: str) -> str:
    return (m or "").strip().lower()

def _method_display(m_key: str) -> str:
    m = _norm_method_key(m_key)
    return DISPLAY_NAME.get(m, m_key)

def _fmt_mean_std(mean: float, std: float, *, scale_100: bool = True, digits: int = 2) -> str:
    if mean is None or std is None:
        return ""
    mean = float(mean)
    std = float(std)
    if scale_100:
        mean *= 100.0
        std *= 100.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"

def _safe_load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _ordered_method_keys(methods_dict: Dict[str, Any]) -> List[str]:
    keys = [_norm_method_key(k) for k in methods_dict.keys()]
    # keep known ones first, then the rest alphabetically
    known = [k for k in METHOD_ORDER if k in keys]
    rest = sorted([k for k in keys if k not in known])
    return known + rest

# -----------------------------
# 1) Distribution plot (2D)
# -----------------------------
def plot_distribution_2d_tfidf(
    *,
    X_text: List[str],
    y: np.ndarray,
    label_names: List[str],
    tfidf_cfg: Dict[str, Any],
    out_png: Path,
    title: str = "2D Distribution (cleaned dataset) — TF-IDF → SVD",
    draw_boundary: bool = False,
    random_state: int = 42,
) -> None:
    vect_cfg = get_dict(tfidf_cfg, "vectorizer", where="features.tfidf")
    trf_cfg = get_dict(tfidf_cfg, "transformer", where="features.tfidf")

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
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
        grid = np.c_[xx.ravel(), yy.ravel()]
        prob = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        plt.contourf(xx, yy, prob, alpha=0.15)
        plt.contour(xx, yy, prob, levels=[0.5], linestyles="--")

    # scatter by class (do NOT force colors)
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
    plt.savefig(out_png, dpi=220)
    plt.close()

# -----------------------------
# 2) Results table (ALL methods, ALL metrics)
# -----------------------------
def build_results_table(
    *,
    cv_metrics: Dict[str, Any],
    out_csv: Path,
    out_tex: Path,
) -> pd.DataFrame:
    methods = cv_metrics.get("methods", {}) or {}

    rows = []
    for m_key in _ordered_method_keys(methods):
        m_val = methods.get(m_key) or {}
        sm = (m_val.get("summary") or {})

        rows.append(
            {
                "Embedding Model": _method_display(m_key),
                "Accuracy (%)": _fmt_mean_std(sm.get("accuracy_mean"), sm.get("accuracy_std")),
                "Precision (%)": _fmt_mean_std(sm.get("precision_mean"), sm.get("precision_std")),
                "Recall (%)": _fmt_mean_std(sm.get("recall_mean"), sm.get("recall_std")),
                "F1-Score (%)": _fmt_mean_std(sm.get("f1_mean"), sm.get("f1_std")),
            }
        )

    df = pd.DataFrame(rows)

    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    ensure_parent(out_tex)
    out_tex.write_text(df.to_latex(index=False), encoding="utf-8")

    return df

# -----------------------------
# 3) Boxplots (per metric, across models)
# -----------------------------
def _extract_fold_metric_arrays(
    cv_metrics: Dict[str, Any],
    metric_key: str,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Returns:
      labels: list of method display names
      data: list of arrays (each is folds values in percent)
    """
    methods = cv_metrics.get("methods", {}) or {}

    labels: List[str] = []
    data: List[np.ndarray] = []

    for m_key in _ordered_method_keys(methods):
        m_val = methods.get(m_key) or {}
        fold_metrics = m_val.get("fold_metrics") or []
        vals = []
        for r in fold_metrics:
            if metric_key in r:
                vals.append(float(r[metric_key]))
        if len(vals) == 0:
            continue
        labels.append(_method_display(m_key))
        data.append(np.array(vals, dtype=float) * 100.0)

    return labels, data

def plot_metric_boxplot(
    *,
    cv_metrics: Dict[str, Any],
    metric_key: str,
    out_png: Path,
    title: Optional[str] = None,
) -> None:
    if metric_key not in METRIC_INFO:
        raise ValueError(f"Unknown metric_key: {metric_key}")

    y_label, _short = METRIC_INFO[metric_key]
    if title is None:
        title = f"{_short} across Cross-Validation Folds"

    labels, data = _extract_fold_metric_arrays(cv_metrics, metric_key=metric_key)

    if len(data) == 0:
        print(f"[WARN] No data for metric={metric_key}; skip: {out_png}")
        return

    plt.figure()
    ax = plt.gca()

    plt.boxplot(data, labels=labels, showfliers=True)

    # mean markers
    means = [float(np.nanmean(x)) for x in data]
    xs = np.arange(1, len(means) + 1)
    plt.scatter(xs, means, marker="^", label="Mean")

    # p-value (Friedman) if possible
    p_val_txt = ""
    try:
        from scipy.stats import friedmanchisquare
        if len(data) >= 2:
            min_len = min(len(x) for x in data)
            arrs = [x[:min_len] for x in data]
            stat, p = friedmanchisquare(*arrs)
            p_val_txt = f"p_val = {p:.2f}"
    except Exception:
        p_val_txt = ""

    plt.title(title)
    plt.ylabel(y_label)

    if p_val_txt:
        plt.legend(title=p_val_txt)
    else:
        plt.legend()

    plt.tight_layout()
    ensure_parent(out_png)
    plt.savefig(out_png, dpi=220)
    plt.close()

# -----------------------------
# main
# -----------------------------
def run_make_report(
    *,
    config_path: Optional[str],
    outdir: str,
    draw_boundary: bool,
) -> None:
    cfg = load_text_config(config_path)
    text_cfg = get_text_config(cfg=cfg)
    feat_cfg = get_features_config(cfg=cfg)

    common_cfg = get_dict(cfg, "common", where="root")
    seed = int(common_cfg.get("seed", 42))

    cleaned_jsonl = Path(str(require(text_cfg, "cleaned_jsonl", where="text")))

    cv_cfg = get_dict(feat_cfg, "crossval", where="features")
    metrics_output = Path(str(require(cv_cfg, "metrics_output", where="features.crossval")))

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load cleaned dataset for distribution
    data = load_cleaned_dataset(cleaned_jsonl)
    X_text = data.X
    y = data.y
    label_names = data.label_names

    # 1) distribution (processed dataset)
    tfidf_cfg = get_dict(feat_cfg, "tfidf", where="features")
    plot_distribution_2d_tfidf(
        X_text=X_text,
        y=y,
        label_names=label_names,
        tfidf_cfg=tfidf_cfg,
        out_png=out_dir / "dist_2d_tfidf.png",
        draw_boundary=bool(draw_boundary),
        random_state=seed,
    )

    # 2/3) metrics-based artifacts
    if not metrics_output.exists():
        print(f"[WARN] metrics_output not found: {metrics_output}")
        print("[WARN] Please run:  python -m paper.evaluate_cv --config <...>")
        return

    cv_metrics = _safe_load_json(metrics_output)

    # 2) total table (ALL methods, ALL metrics)
    build_results_table(
        cv_metrics=cv_metrics,
        out_csv=out_dir / "results_table.csv",
        out_tex=out_dir / "results_table.tex",
    )

    # 3) boxplots for 4 metrics
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="accuracy",
        out_png=out_dir / "boxplot_accuracy.png",
        title="Accuracy across Cross-Validation Folds",
    )
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="precision",
        out_png=out_dir / "boxplot_precision.png",
        title="Precision across Cross-Validation Folds",
    )
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="recall",
        out_png=out_dir / "boxplot_recall.png",
        title="Recall across Cross-Validation Folds",
    )
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="f1",
        out_png=out_dir / "boxplot_f1.png",
        title="F1-Score across Cross-Validation Folds",
    )

    print(f"[INFO] Report outputs saved to: {out_dir}")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate distribution plot + results table + boxplots (4 metrics).")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument("--outdir", type=str, default="/content/chinese_combined/report", help="Output directory")
    p.add_argument("--draw-boundary", action="store_true", help="Draw decision boundary on the distribution plot.")
    return p

def cli_main() -> None:
    args, _unknown = build_arg_parser().parse_known_args()  # ignore Colab injected args like -f
    run_make_report(
        config_path=args.config,
        outdir=args.outdir,
        draw_boundary=bool(args.draw_boundary),
    )

if __name__ == "__main__":
    cli_main()
