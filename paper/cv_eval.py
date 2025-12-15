# -*- coding: utf-8 -*-
"""
paper/cv_eval.py

Evaluation core:
- build LogisticRegression from YAML
- evaluate with precomputed folds
- (optional) TF-IDF train-only per fold evaluation (to keep evaluate_cv clean)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


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
) -> Dict[str, Any]:
    clf_template = build_logreg(logreg_cfg)

    fold_rows: List[Dict[str, Any]] = []
    cms: List[List[List[int]]] = []

    for fold_i, (tr, te) in enumerate(folds, start=1):
        Xtr = X_feats_all[tr]
        Xte = X_feats_all[te]
        ytr = y[tr]
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
                "fold": fold_i,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
            }
        )

        if print_report:
            print(f"\nFold {fold_i} ({method_name}):")
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
    }

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
) -> Dict[str, Any]:
    """
    TF-IDF strictly fit on TRAIN per fold.
    Moved here to keep paper/evaluate_cv.py clean.
    """
    from paper.cv_features import tfidf_features_fold_fit  # local import keeps dependencies clean

    clf_template = build_logreg(logreg_cfg)

    fold_rows: List[Dict[str, Any]] = []
    cms: List[List[List[int]]] = []

    for fold_i, (tr, te) in enumerate(folds, start=1):
        # folds store numpy arrays already, but be defensive
        tr_idx = tr.tolist() if hasattr(tr, "tolist") else list(tr)
        te_idx = te.tolist() if hasattr(te, "tolist") else list(te)

        Xtr_txt = [X_text[i] for i in tr_idx]
        Xte_txt = [X_text[i] for i in te_idx]

        Xtr, Xte = tfidf_features_fold_fit(
            Xtr_txt,
            Xte_txt,
            vectorizer_cfg=vectorizer_cfg,
            transformer_cfg=transformer_cfg,
        )

        # IMPORTANT: index y with the SAME index type (list of ints)
        ytr = y[tr_idx]
        yte = y[te_idx]

        clf = LogisticRegression(**clf_template.get_params())
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        acc = float(metrics.accuracy_score(yte, ypred))
        prec = float(metrics.precision_score(yte, ypred, average=average, zero_division=zero_division))
        rec = float(metrics.recall_score(yte, ypred, average=average, zero_division=zero_division))
        f1 = float(metrics.f1_score(yte, ypred, average=average, zero_division=zero_division))

        fold_rows.append(
            {
                "fold": fold_i,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
            }
        )

        if print_report:
            print(f"\nFold {fold_i} ({method_name} train-only):")
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
    }
