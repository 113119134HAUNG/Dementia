# -*- coding: utf-8 -*-
"""
paper/cv_eval.py

Evaluation core:
- build LogisticRegression from YAML
- evaluate with precomputed folds
- (optional) TF-IDF train-only per fold evaluation
- (optional) TRAIN-only balancing per fold (downsample/upsample)

Notes:
- For TF-IDF "train-only", we MUST:
  (1) (optional) balance TRAIN indices
  (2) fit TF-IDF on TRAIN only
  (3) transform TEST using TRAIN-fitted TF-IDF
  This avoids leakage and matches paper-strict setting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# ---------------------------
# Model builder
# ---------------------------
def build_logreg(cfg: Dict[str, Any]) -> LogisticRegression:
    C = float(cfg.get("C", 1.0))
    class_weight = cfg.get("class_weight", "balanced")
    solver = cfg.get("solver", "lbfgs")
    max_iter = int(cfg.get("max_iter", 1000))
    random_state = cfg.get("random_state", None)
    if random_state is not None:
        random_state = int(random_state)

    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )


# ---------------------------
# Train balancing helpers
# ---------------------------
def _parse_train_balance_cfg(train_balance_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = train_balance_cfg if isinstance(train_balance_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    strategy = str(cfg.get("strategy", "none")).strip().lower()
    if strategy not in ("none", "downsample", "upsample"):
        raise ValueError("features.crossval.train_balance.strategy must be one of: none, downsample, upsample")
    rs = cfg.get("random_state", None)
    rs_i = None if rs is None else int(rs)
    return {"enabled": enabled, "strategy": strategy, "random_state": rs_i}


def _balance_train_indices(
    tr: np.ndarray,
    y: np.ndarray,
    *,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return new training indices (subset or resampled) for TRAIN ONLY.
    - downsample: each class -> min class count (no replacement)
    - upsample:   each class -> max class count (with replacement)
    """
    tr = np.asarray(tr, dtype=np.int64)
    ytr = y[tr]
    classes = np.unique(ytr)
    if len(classes) < 2:
        return tr

    idx_by_class = {c: np.flatnonzero(ytr == c) for c in classes}
    counts = {int(c): int(len(v)) for c, v in idx_by_class.items()}
    if any(v == 0 for v in counts.values()):
        return tr

    if strategy == "downsample":
        n = min(counts.values())
        picked_pos = []
        for c in classes:
            pos = idx_by_class[c]
            sel = pos if len(pos) == n else rng.choice(pos, size=n, replace=False)
            picked_pos.append(sel)
        picked_pos = np.concatenate(picked_pos, axis=0)
        rng.shuffle(picked_pos)
        return tr[picked_pos]

    if strategy == "upsample":
        n = max(counts.values())
        picked_pos = []
        for c in classes:
            pos = idx_by_class[c]
            sel = pos if len(pos) == n else rng.choice(pos, size=n, replace=True)
            picked_pos.append(sel)
        picked_pos = np.concatenate(picked_pos, axis=0)
        rng.shuffle(picked_pos)
        return tr[picked_pos]

    return tr


def _label_counts(y: np.ndarray, idx: np.ndarray) -> Dict[str, int]:
    """
    Counts for labels 0/1 only (your pipeline uses 0=HC, 1=AD).
    """
    idx = np.asarray(idx, dtype=np.int64)
    yy = y[idx]
    return {"HC": int((yy == 0).sum()), "AD": int((yy == 1).sum())}


# ---------------------------
# Generic evaluation: precomputed features
# ---------------------------
def evaluate_with_precomputed_folds(
    X_feats_all: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    logreg_cfg: Dict[str, Any],
    average: str,
    zero_division: int,
    print_report: bool,
    print_cm: bool,
    label_names: List[str],
    method_name: str,
    train_balance_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    clf_template = build_logreg(logreg_cfg)

    tb = _parse_train_balance_cfg(train_balance_cfg)
    do_balance = bool(tb["enabled"]) and tb["strategy"] != "none"

    fold_rows: List[Dict[str, Any]] = []
    cms: List[List[List[int]]] = []

    for fold_i, (tr, te) in enumerate(folds, start=1):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        tr_use = tr
        n_train_before = int(len(tr))

        if do_balance:
            seed = (tb["random_state"] if tb["random_state"] is not None else 0) + int(fold_i)
            rng = np.random.default_rng(seed)
            tr_use = _balance_train_indices(tr, y, strategy=tb["strategy"], rng=rng)

        Xtr = X_feats_all[tr_use]
        Xte = X_feats_all[te]
        ytr = y[tr_use]
        yte = y[te]

        clf = LogisticRegression(**clf_template.get_params())
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        acc = float(metrics.accuracy_score(yte, ypred))
        prec = float(metrics.precision_score(yte, ypred, average=average, zero_division=zero_division))
        rec = float(metrics.recall_score(yte, ypred, average=average, zero_division=zero_division))
        f1 = float(metrics.f1_score(yte, ypred, average=average, zero_division=zero_division))

        fold_rows.append(
            {
                "fold": int(fold_i),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_train": int(len(tr_use)),
                "n_train_before_balance": n_train_before,
                "n_test": int(len(te)),
                "train_label_counts": _label_counts(y, tr_use),
                "test_label_counts": _label_counts(y, te),
            }
        )

        if print_report:
            suffix = f" ({tb['strategy']} train-balance)" if do_balance else ""
            print(f"\nFold {fold_i} ({method_name}{suffix}):")
            print(
                metrics.classification_report(
                    yte,
                    ypred,
                    target_names=label_names,
                    zero_division=zero_division,
                )
            )

        if print_cm:
            cm = metrics.confusion_matrix(yte, ypred, labels=[0, 1]).astype(int).tolist()
            cms.append(cm)
            print(f"[Confusion Matrix] Fold {fold_i} ({method_name}) labels={label_names}: {cm}")

    arr_acc = np.array([r["accuracy"] for r in fold_rows], dtype=np.float64)
    arr_prec = np.array([r["precision"] for r in fold_rows], dtype=np.float64)
    arr_rec = np.array([r["recall"] for r in fold_rows], dtype=np.float64)
    arr_f1 = np.array([r["f1"] for r in fold_rows], dtype=np.float64)

    summary = {
        "accuracy_mean": float(arr_acc.mean()),
        "accuracy_std": float(arr_acc.std()),
        "precision_mean": float(arr_prec.mean()),
        "precision_std": float(arr_prec.std()),
        "recall_mean": float(arr_rec.mean()),
        "recall_std": float(arr_rec.std()),
        "f1_mean": float(arr_f1.mean()),
        "f1_std": float(arr_f1.std()),
    }

    return {
        "method": method_name,
        "fold_metrics": fold_rows,
        "summary": summary,
        "confusion_matrices": cms if print_cm else [],
        "train_balance": tb,
    }


# ---------------------------
# TF-IDF train-only (per fold)
# ---------------------------
def evaluate_tfidf_trainonly(
    *,
    X_text: List[str],
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    vectorizer_cfg: Dict[str, Any],
    transformer_cfg: Dict[str, Any],
    logreg_cfg: Dict[str, Any],
    average: str,
    zero_division: int,
    print_report: bool,
    print_cm: bool,
    label_names: List[str],
    method_name: str = "tfidf",
    train_balance_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    TF-IDF strictly fit on TRAIN per fold.
    Also supports TRAIN-only balancing per fold (before TF-IDF fit).
    """
    from paper.cv_features import tfidf_features_fold_fit  # local import keeps deps clean

    clf_template = build_logreg(logreg_cfg)

    tb = _parse_train_balance_cfg(train_balance_cfg)
    do_balance = bool(tb["enabled"]) and tb["strategy"] != "none"

    fold_rows: List[Dict[str, Any]] = []
    cms: List[List[List[int]]] = []

    for fold_i, (tr, te) in enumerate(folds, start=1):
        tr = np.asarray(tr, dtype=np.int64)
        te = np.asarray(te, dtype=np.int64)

        n_train_before = int(len(tr))
        tr_use = tr

        if do_balance:
            seed = (tb["random_state"] if tb["random_state"] is not None else 0) + int(fold_i)
            rng = np.random.default_rng(seed)
            tr_use = _balance_train_indices(tr, y, strategy=tb["strategy"], rng=rng)

        # Text slices
        Xtr_txt = [X_text[i] for i in tr_use.tolist()]
        Xte_txt = [X_text[i] for i in te.tolist()]

        # Fit TF-IDF on TRAIN (after balancing), transform TEST
        Xtr, Xte = tfidf_features_fold_fit(
            Xtr_txt,
            Xte_txt,
            vectorizer_cfg=vectorizer_cfg,
            transformer_cfg=transformer_cfg,
        )

        ytr = y[tr_use]
        yte = y[te]

        clf = LogisticRegression(**clf_template.get_params())
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        acc = float(metrics.accuracy_score(yte, ypred))
        prec = float(metrics.precision_score(yte, ypred, average=average, zero_division=zero_division))
        rec = float(metrics.recall_score(yte, ypred, average=average, zero_division=zero_division))
        f1 = float(metrics.f1_score(yte, ypred, average=average, zero_division=zero_division))

        fold_rows.append(
            {
                "fold": int(fold_i),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_train": int(len(tr_use)),
                "n_train_before_balance": n_train_before,
                "n_test": int(len(te)),
                "train_label_counts": _label_counts(y, tr_use),
                "test_label_counts": _label_counts(y, te),
            }
        )

        if print_report:
            suffix = f" ({tb['strategy']} train-balance)" if do_balance else ""
            print(f"\nFold {fold_i} ({method_name} train-only{suffix}):")
            print(metrics.classification_report(yte, ypred, target_names=label_names, zero_division=zero_division))

        if print_cm:
            cm = metrics.confusion_matrix(yte, ypred, labels=[0, 1]).astype(int).tolist()
            cms.append(cm)
            print(f"[Confusion Matrix] Fold {fold_i} ({method_name} train-only) labels={label_names}: {cm}")

    arr_acc = np.array([r["accuracy"] for r in fold_rows], dtype=np.float64)
    arr_prec = np.array([r["precision"] for r in fold_rows], dtype=np.float64)
    arr_rec = np.array([r["recall"] for r in fold_rows], dtype=np.float64)
    arr_f1 = np.array([r["f1"] for r in fold_rows], dtype=np.float64)

    summary = {
        "accuracy_mean": float(arr_acc.mean()),
        "accuracy_std": float(arr_acc.std()),
        "precision_mean": float(arr_prec.mean()),
        "precision_std": float(arr_prec.std()),
        "recall_mean": float(arr_rec.mean()),
        "recall_std": float(arr_rec.std()),
        "f1_mean": float(arr_f1.mean()),
        "f1_std": float(arr_f1.std()),
    }

    return {
        "method": method_name,
        "fold_metrics": fold_rows,
        "summary": summary,
        "confusion_matrices": cms if print_cm else [],
        "tfidf_fit_scope": "train",
        "train_balance": tb,
    }