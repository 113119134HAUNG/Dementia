# -*- coding: utf-8 -*-
"""
paper/make_report.py

Generate paper-style artifacts from existing outputs (paper-strict):
Inputs:
- cleaned.jsonl (from preprocess.preprocess_chinese)
- cv_metrics.json (from paper.evaluate_cv)

Outputs (default outdir=/content/chinese_combined/report):
1) dist_2d_tfidf_cleaned.png                  (post-clean dataset)
2) dist_2d_tfidf_combined_preclean.png        (optional; pre-clean combined dataset if exists)
3) length_dist_cleaned.png                    (text-length distribution; before vectorization; post-clean)
4) length_dist_combined_preclean.png          (optional; before vectorization; pre-clean)
5) results_table.csv + results_table.tex
6) boxplot_accuracy.png / boxplot_precision.png / boxplot_recall.png / boxplot_f1.png
7) line_accuracy_per_fold.png / line_precision_per_fold.png / line_recall_per_fold.png / line_f1_per_fold.png

Works in Colab/Jupyter (ignores injected argv like -f).

Notes:
- 2D distribution uses TF-IDF -> TruncatedSVD(2) (visualization only).
- draw_boundary draws LR decision boundary in the 2D SVD space (visualization only).
- boundary_scope controls where boundary is drawn: cleaned only (default) or both cleaned+combined.
- "Before vectorization" distribution is shown via text length plots (no TF-IDF).
- Method order fixed to match YAML defaults: tfidf, bert, glove, gemma, linq.
- ALL titles/axes are English only (paper-ready).
"""

from __future__ import annotations

import argparse
import json
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
METHOD_ORDER = ["tfidf", "bert", "glove", "gemma", "linq"]

DISPLAY_NAME = {
    "bert": "BERT-base (mean pooling)",
    "tfidf": "TF-IDF (char 2–4gram)",
    "glove": "fastText/GloVe (300d)",
    "gemma": "Gemma-2B (mean pooling)",
    "linq": "Linq-Embed-Mistral (mean pooling)",
}

METRIC_INFO = {
    "accuracy": ("Accuracy (%)", "Accuracy"),
    "precision": ("Precision (%)", "Precision"),
    "recall": ("Recall (%)", "Recall"),
    "f1": ("F1-Score (%)", "F1-Score"),
}


def _norm_method_key(m: str) -> str:
    return (m or "").strip().lower()


def _method_display(m_key: str) -> str:
    m = _norm_method_key(m_key)
    return DISPLAY_NAME.get(m, m)


def _fmt_mean_std(mean: Any, std: Any, *, scale_100: bool = True, digits: int = 2) -> str:
    if mean is None or std is None:
        return ""
    mean_f = float(mean)
    std_f = float(std)
    if scale_100:
        mean_f *= 100.0
        std_f *= 100.0
    return f"{mean_f:.{digits}f} ± {std_f:.{digits}f}"


def _safe_load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_method_keys(methods_dict: Dict[str, Any]) -> List[str]:
    keys = [_norm_method_key(k) for k in (methods_dict or {}).keys()]
    known = [k for k in METHOD_ORDER if k in keys]
    rest = sorted([k for k in keys if k not in known])
    return known + rest


# -----------------------------
# lightweight JSONL loader (for combined/pre-clean)
# -----------------------------
def _load_text_label_from_jsonl(
    path: Path,
    *,
    text_cols: Optional[List[str]] = None,
    label_col: str = "Diagnosis",
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Generic loader for jsonl where:
      - label_col (default 'Diagnosis') holds labels like 'HC','AD','MCI'...
      - text is in one of text_cols candidates
    Returns:
      X_text, y_int (0..K-1), label_names (ordered by sorted label strings)
    """
    if text_cols is None:
        text_cols = ["Text_interviewer_participant", "text", "Text", "utterance", "sentence"]

    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    df = pd.read_json(path, lines=True)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in: {path}")

    text_col_used = None
    for c in text_cols:
        if c in df.columns:
            text_col_used = c
            break
    if text_col_used is None:
        raise ValueError(f"No text column found in {path}. Tried: {text_cols}")

    labels_raw = df[label_col].astype(str).fillna("").tolist()
    texts_raw = df[text_col_used].fillna("").astype(str).tolist()

    keep = [i for i, t in enumerate(texts_raw) if len(t.strip()) > 0]
    labels = [labels_raw[i] for i in keep]
    texts = [texts_raw[i] for i in keep]

    uniq = sorted(set(labels))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y = np.array([mapping[lab] for lab in labels], dtype=int)

    return texts, y, uniq


# -----------------------------
# color utilities
# -----------------------------
def _gradient_colors(n: int, cmap_name: str = "viridis", alpha: float = 0.55) -> List[Tuple[float, float, float, float]]:
    n = max(1, int(n))
    cmap = plt.get_cmap(cmap_name)
    xs = np.linspace(0.25, 0.90, n)
    cols = [list(cmap(x)) for x in xs]
    for c in cols:
        c[3] = alpha
    return [tuple(c) for c in cols]


# -----------------------------
# 0) Length distribution (pre-vectorization view)
# -----------------------------
def plot_length_distribution(
    *,
    X_text: List[str],
    y: np.ndarray,
    label_names: List[str],
    out_png: Path,
    title: str,
) -> None:
    lens = np.array([len((s or "")) for s in X_text], dtype=float)

    groups: List[np.ndarray] = []
    labels: List[str] = []
    for k in sorted(np.unique(y)):
        mask = (y == k)
        name = label_names[int(k)] if int(k) < len(label_names) else f"class_{k}"
        groups.append(lens[mask])
        labels.append(name)

    plt.figure(figsize=(8.8, 5.2))
    ax = plt.gca()
    bp = plt.boxplot(groups, labels=labels, showfliers=True, patch_artist=True)

    cols = _gradient_colors(len(groups), cmap_name="viridis", alpha=0.40)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_edgecolor((0, 0, 0, 0.65))
        patch.set_linewidth(1.1)

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Text length (characters)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    ensure_parent(out_png)
    plt.savefig(out_png, dpi=240)
    plt.close()


# -----------------------------
# 1) Distribution plot (2D): TF-IDF -> SVD
# -----------------------------
def plot_distribution_2d_tfidf(
    *,
    X_text: List[str],
    y: np.ndarray,
    label_names: List[str],
    tfidf_cfg: Dict[str, Any],
    out_png: Path,
    title: str,
    draw_boundary: bool = False,
    random_state: int = 42,
) -> None:
    vect_cfg = get_dict(tfidf_cfg, "vectorizer", where="features.tfidf")
    trf_cfg = get_dict(tfidf_cfg, "transformer", where="features.tfidf")

    X_all = tfidf_features_all(X_text, vectorizer_cfg=vect_cfg, transformer_cfg=trf_cfg)

    svd = TruncatedSVD(n_components=2, random_state=random_state)
    X2 = svd.fit_transform(X_all)

    evr = getattr(svd, "explained_variance_ratio_", None)
    if evr is None or len(evr) < 2:
        evr = [np.nan, np.nan]
    evr = np.array(evr, dtype=float)
    evr_sum = float(np.nansum(evr)) if np.isfinite(evr).any() else float("nan")

    xlab = "TruncatedSVD component 1"
    ylab = "TruncatedSVD component 2"
    if np.isfinite(evr[0]):
        xlab = f"TruncatedSVD component 1 (explained variance {evr[0]*100:.1f}%)"
    if np.isfinite(evr[1]):
        ylab = f"TruncatedSVD component 2 (explained variance {evr[1]*100:.1f}%)"

    title2 = title
    if np.isfinite(evr_sum):
        title2 = f"{title}\n(Explained variance by 2 components: {evr_sum*100:.1f}%)"

    plt.figure(figsize=(9.2, 6.2))
    ax = plt.gca()

    if draw_boundary:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        clf.fit(X2, y)

        x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
        y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 280), np.linspace(y_min, y_max, 280))
        grid = np.c_[xx.ravel(), yy.ravel()]
        prob = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

        plt.contourf(xx, yy, prob, alpha=0.14)
        plt.contour(xx, yy, prob, levels=[0.5], linestyles="--", linewidths=1.2)
        ax.text(
            0.01,
            0.01,
            "Logistic regression decision boundary in 2D (visualization only)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
        )

    for k in sorted(np.unique(y)):
        mask = (y == k)
        name = label_names[int(k)] if int(k) < len(label_names) else f"class_{k}"
        plt.scatter(X2[mask, 0], X2[mask, 1], label=name, s=26, alpha=0.82)

    plt.title(title2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax.grid(True, linestyle="--", alpha=0.25)
    plt.legend(title="Class", frameon=True)
    plt.tight_layout()

    ensure_parent(out_png)
    plt.savefig(out_png, dpi=240)
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
                "Model": _method_display(m_key),
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
    out_tex.write_text(df.to_latex(index=False, escape=False), encoding="utf-8")

    return df


# -----------------------------
# 3) Metrics plots: boxplots + fold-lines
# -----------------------------
def _extract_fold_metric_arrays(
    cv_metrics: Dict[str, Any],
    metric_key: str,
) -> Tuple[List[str], List[np.ndarray]]:
    methods = cv_metrics.get("methods", {}) or {}

    labels: List[str] = []
    data: List[np.ndarray] = []

    for m_key in _ordered_method_keys(methods):
        m_val = methods.get(m_key) or {}
        fold_metrics = m_val.get("fold_metrics") or []
        vals = []
        for r in fold_metrics:
            if isinstance(r, dict) and metric_key in r:
                vals.append(float(r[metric_key]))
        if len(vals) == 0:
            continue
        labels.append(_method_display(m_key))
        data.append(np.array(vals, dtype=float) * 100.0)

    return labels, data


def _friedman_pvalue_text(data: List[np.ndarray]) -> str:
    try:
        from scipy.stats import friedmanchisquare  # type: ignore
    except Exception:
        return ""

    if len(data) < 2:
        return ""

    min_len = min(len(x) for x in data)
    if min_len < 2:
        return ""

    arrs = [x[:min_len] for x in data]
    try:
        _stat, p = friedmanchisquare(*arrs)
        return f"Friedman p = {p:.3f}"
    except Exception:
        return ""


def plot_metric_boxplot(
    *,
    cv_metrics: Dict[str, Any],
    metric_key: str,
    out_png: Path,
    title: Optional[str] = None,
    cmap_name: str = "viridis",
) -> None:
    if metric_key not in METRIC_INFO:
        raise ValueError(f"Unknown metric_key: {metric_key}")

    y_label, short = METRIC_INFO[metric_key]
    if title is None:
        title = f"{short} across cross-validation folds"

    labels, data = _extract_fold_metric_arrays(cv_metrics, metric_key=metric_key)
    if len(data) == 0:
        print(f"[WARN] No data for metric={metric_key}; skip: {out_png}")
        return

    colors = _gradient_colors(len(data), cmap_name=cmap_name, alpha=0.55)

    plt.figure(figsize=(10.6, 5.4))
    ax = plt.gca()

    bp = plt.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
        medianprops=dict(linewidth=1.7),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor((0, 0, 0, 0.65))

    means = [float(np.nanmean(x)) for x in data]
    xs = np.arange(1, len(means) + 1)
    plt.scatter(xs, means, marker="D", s=46, label="Mean")

    plt.title(title)
    plt.xlabel("Embedding method")
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(rotation=12, ha="right")

    ptxt = _friedman_pvalue_text(data)
    if ptxt:
        plt.legend(title=ptxt, frameon=True)
    else:
        plt.legend(frameon=True)

    plt.tight_layout()
    ensure_parent(out_png)
    plt.savefig(out_png, dpi=240)
    plt.close()


def plot_metric_fold_lines(
    *,
    cv_metrics: Dict[str, Any],
    metric_key: str,
    out_png: Path,
    title: Optional[str] = None,
) -> None:
    if metric_key not in METRIC_INFO:
        raise ValueError(f"Unknown metric_key: {metric_key}")

    y_label, short = METRIC_INFO[metric_key]
    if title is None:
        title = f"{short} by fold (stability)"

    methods = cv_metrics.get("methods", {}) or {}
    ordered = _ordered_method_keys(methods)

    series = []
    labels = []
    min_len = None
    for m_key in ordered:
        fold_metrics = (methods.get(m_key) or {}).get("fold_metrics") or []
        vals = [float(r[metric_key]) * 100.0 for r in fold_metrics if isinstance(r, dict) and metric_key in r]
        if len(vals) == 0:
            continue
        min_len = len(vals) if min_len is None else min(min_len, len(vals))
        series.append(vals)
        labels.append(_method_display(m_key))

    if not series or min_len is None or min_len < 2:
        print(f"[WARN] Not enough data for fold-line metric={metric_key}; skip: {out_png}")
        return

    series = [np.array(v[:min_len], dtype=float) for v in series]
    folds = np.arange(1, min_len + 1)

    plt.figure(figsize=(10.6, 5.2))
    ax = plt.gca()

    for lab, arr in zip(labels, series):
        plt.plot(folds, arr, marker="o", linewidth=1.5, label=lab)

    plt.title(title)
    plt.xlabel("Fold index")
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(folds)

    plt.legend(title="Embedding method", frameon=True)
    plt.tight_layout()
    ensure_parent(out_png)
    plt.savefig(out_png, dpi=240)
    plt.close()


# -----------------------------
# main
# -----------------------------
def run_make_report(
    *,
    config_path: Optional[str],
    outdir: str,
    draw_boundary: bool,
    boundary_scope: str,
    skip_combined: bool,
) -> None:
    boundary_scope = (boundary_scope or "cleaned").strip().lower()
    if boundary_scope not in ("cleaned", "both"):
        raise ValueError("boundary_scope must be one of: cleaned | both")

    cfg = load_text_config(config_path)
    text_cfg = get_text_config(cfg=cfg)
    feat_cfg = get_features_config(cfg=cfg)

    common_cfg = get_dict(cfg, "common", where="root")
    seed = int(common_cfg.get("seed", 42))

    cleaned_jsonl = Path(str(require(text_cfg, "cleaned_jsonl", where="text")))

    output_dir = Path(str(require(text_cfg, "output_dir", where="text")))
    combined_name = str(require(text_cfg, "combined_name", where="text"))
    combined_jsonl = output_dir / combined_name

    cv_cfg = get_dict(feat_cfg, "crossval", where="features")
    metrics_output = Path(str(require(cv_cfg, "metrics_output", where="features.crossval")))

    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfidf_cfg = get_dict(feat_cfg, "tfidf", where="features")

    data = load_cleaned_dataset(cleaned_jsonl)
    X_text = data.X
    y = data.y
    label_names = data.label_names

    plot_length_distribution(
        X_text=X_text,
        y=y,
        label_names=label_names,
        out_png=out_dir / "length_dist_cleaned.png",
        title="Text length distribution (post-clean dataset)",
    )

    draw_cleaned = bool(draw_boundary)
    plot_distribution_2d_tfidf(
        X_text=X_text,
        y=y,
        label_names=label_names,
        tfidf_cfg=tfidf_cfg,
        out_png=out_dir / "dist_2d_tfidf_cleaned.png",
        title="2D distribution (post-clean dataset): TF-IDF → TruncatedSVD(2)",
        draw_boundary=draw_cleaned,
        random_state=seed,
    )

    if (not skip_combined) and combined_jsonl.exists():
        try:
            Xc, yc, label_names_c = _load_text_label_from_jsonl(combined_jsonl)

            plot_length_distribution(
                X_text=Xc,
                y=yc,
                label_names=label_names_c,
                out_png=out_dir / "length_dist_combined_preclean.png",
                title="Text length distribution (pre-clean combined dataset)",
            )

            draw_combined = bool(draw_boundary) and (boundary_scope == "both")
            plot_distribution_2d_tfidf(
                X_text=Xc,
                y=yc,
                label_names=label_names_c,
                tfidf_cfg=tfidf_cfg,
                out_png=out_dir / "dist_2d_tfidf_combined_preclean.png",
                title="2D distribution (pre-clean combined dataset): TF-IDF → TruncatedSVD(2)",
                draw_boundary=draw_combined,
                random_state=seed,
            )
        except Exception as e:
            print(f"[WARN] Skip pre-clean combined plots due to error: {type(e).__name__}: {e}")

    if not metrics_output.exists():
        print(f"[WARN] metrics_output not found: {metrics_output}")
        print("[WARN] Please run:  python -m paper.evaluate_cv --config <...>")
        return

    cv_metrics = json.loads(metrics_output.read_text(encoding="utf-8"))

    build_results_table(
        cv_metrics=cv_metrics,
        out_csv=out_dir / "results_table.csv",
        out_tex=out_dir / "results_table.tex",
    )

    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="accuracy",
        out_png=out_dir / "boxplot_accuracy.png",
        title="Accuracy across cross-validation folds",
        cmap_name="viridis",
    )
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="precision",
        out_png=out_dir / "boxplot_precision.png",
        title="Precision across cross-validation folds",
        cmap_name="viridis",
    )
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="recall",
        out_png=out_dir / "boxplot_recall.png",
        title="Recall across cross-validation folds",
        cmap_name="viridis",
    )
    plot_metric_boxplot(
        cv_metrics=cv_metrics,
        metric_key="f1",
        out_png=out_dir / "boxplot_f1.png",
        title="F1-Score across cross-validation folds",
        cmap_name="viridis",
    )

    for mk in ["accuracy", "precision", "recall", "f1"]:
        plot_metric_fold_lines(
            cv_metrics=cv_metrics,
            metric_key=mk,
            out_png=out_dir / f"line_{mk}_per_fold.png",
            title=None,
        )

    print(f"[INFO] Report outputs saved to: {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate dist plots + results table + metrics plots.")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument("--outdir", type=str, default="/content/chinese_combined/report", help="Output directory")

    p.add_argument("--draw-boundary", action="store_true")
    p.add_argument("--boundary-scope", type=str, default="cleaned", choices=["cleaned", "both"])
    p.add_argument("--skip-combined", action="store_true")
    return p


def cli_main() -> None:
    args, _unknown = build_arg_parser().parse_known_args()
    run_make_report(
        config_path=args.config,
        outdir=args.outdir,
        draw_boundary=bool(args.draw_boundary),
        boundary_scope=str(args.boundary_scope),
        skip_combined=bool(args.skip_combined),
    )


if __name__ == "__main__":
    cli_main()