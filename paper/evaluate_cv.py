# -*- coding: utf-8 -*-
"""
paper/evaluate_cv.py

This file only:
- reads YAML
- loads cleaned JSONL
- builds/reuses fold indices
- runs methods (tfidf/bert/glove/gemma) using same folds
- saves metrics JSON

No feature extraction code here.
No fold IO code here.
No evaluation-core code here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tools.config_utils import load_text_config, get_text_config, get_features_config

from paper.cv_utils import require, get_dict, norm_str_list, ensure_parent
from paper.cv_dataset import load_cleaned_dataset
from paper.cv_folds import build_folds_indices, save_folds_indices, load_folds_indices
from paper.cv_features import (
    tfidf_features_all,
    bert_embeddings_all,
    glove_embeddings_all,
    gemma_embeddings_all,
)
from paper.cv_eval import (
    evaluate_with_precomputed_folds,
    evaluate_tfidf_trainonly,
)

def run_evaluate_cv(
    config_path: Optional[str] = None,
    *,
    methods_override: Optional[List[str]] = None,
    reuse_folds: bool = False,
) -> str:
    cfg = load_text_config(config_path)
    text_cfg = get_text_config(cfg=cfg)
    feat_cfg = get_features_config(cfg=cfg)

    cv_cfg = get_dict(feat_cfg, "crossval", where="features")

    if not bool(cv_cfg.get("enabled", True)):
        print("[INFO] features.crossval.enabled=false -> skip evaluate_cv.")
        return ""

    # dataset (already cleaned)
    cleaned_jsonl = Path(str(require(text_cfg, "cleaned_jsonl", where="text")))
    data = load_cleaned_dataset(cleaned_jsonl)
    X_text = data.X
    y = data.y
    label_names = data.label_names

    # CV knobs
    n_splits = int(require(cv_cfg, "n_splits", where="features.crossval"))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(require(cv_cfg, "random_state", where="features.crossval"))
    output_indices = Path(str(require(cv_cfg, "output_indices", where="features.crossval")))

    # metrics knobs
    average = str(cv_cfg.get("metrics_average", "macro"))
    zero_division = int(cv_cfg.get("zero_division", 0))
    metrics_output = Path(str(require(cv_cfg, "metrics_output", where="features.crossval")))
    print_report = bool(cv_cfg.get("print_classification_report", True))
    print_cm = bool(cv_cfg.get("print_confusion_matrix", True))
    print_method_summary = bool(cv_cfg.get("print_method_summary", True))

    # method selection
    methods_yaml = norm_str_list(cv_cfg.get("methods")) or ["tfidf", "bert", "glove", "gemma"]
    methods = methods_override if methods_override is not None else methods_yaml

    # feasibility check
    vc = pd.Series(y).value_counts()
    if (vc < n_splits).any():
        raise ValueError(f"Not enough samples per class for {n_splits}-fold CV. Counts:\n{vc}")

    # folds (save once, reuse across methods)
    if reuse_folds:
        folds = load_folds_indices(output_indices)
        print(f"[INFO] Reusing folds from: {output_indices} (n_folds={len(folds)})")
    else:
        folds = build_folds_indices(y, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        save_folds_indices(folds, output_path=output_indices)
        print(f"[INFO] Saved fold indices to: {output_indices} (n_folds={len(folds)})")

    # runtime knobs
    default_bs = int(cv_cfg.get("batch_size", 8))
    bert_bs = int(cv_cfg.get("bert_batch_size", default_bs))
    gemma_bs = int(cv_cfg.get("gemma_batch_size", default_bs))

    tfidf_fit_scope = str(cv_cfg.get("tfidf_fit_scope", "full")).strip().lower()
    if tfidf_fit_scope not in ("full", "train"):
        raise ValueError("features.crossval.tfidf_fit_scope must be 'full' or 'train'.")

    # results skeleton
    results: Dict[str, Any] = {
        "data": {
            "cleaned_jsonl": str(cleaned_jsonl),
            "n_samples": int(len(y)),
            "label_counts": {"HC": int((y == 0).sum()), "AD": int((y == 1).sum())},
        },
        "crossval": {
            "n_splits": int(n_splits),
            "shuffle": bool(shuffle),
            "random_state": int(random_state),
            "output_indices": str(output_indices),
            "metrics_average": average,
            "zero_division": int(zero_division),
            "tfidf_fit_scope": tfidf_fit_scope,
        },
        "methods": {},
    }

    for method in methods:
        m = str(method).strip().lower()
        if not m:
            continue

        if m == "tfidf":
            tfidf_cfg = get_dict(feat_cfg, "tfidf", where="features")
            vect_cfg = get_dict(tfidf_cfg, "vectorizer", where="features.tfidf")
            trf_cfg = get_dict(tfidf_cfg, "transformer", where="features.tfidf")
            logreg_cfg = get_dict(tfidf_cfg, "logreg", where="features.tfidf")

            if tfidf_fit_scope == "full":
                X_all = tfidf_features_all(X_text, vectorizer_cfg=vect_cfg, transformer_cfg=trf_cfg)
                res = evaluate_with_precomputed_folds(
                    X_all,
                    y,
                    folds,
                    logreg_cfg=logreg_cfg,
                    average=average,
                    zero_division=zero_division,
                    print_report=print_report,
                    print_cm=print_cm,
                    label_names=label_names,
                    method_name="tfidf",
                )
            else:
                res = evaluate_tfidf_trainonly(
                    X_text=X_text,
                    y=y,
                    folds=folds,
                    vectorizer_cfg=vect_cfg,
                    transformer_cfg=trf_cfg,
                    logreg_cfg=logreg_cfg,
                    average=average,
                    zero_division=zero_division,
                    print_report=print_report,
                    print_cm=print_cm,
                    label_names=label_names,
                    method_name="tfidf",
                )

            results["methods"]["tfidf"] = res

        elif m == "bert":
            bert_cfg = get_dict(feat_cfg, "bert", where="features")
            model_name = str(require(bert_cfg, "model_name", where="features.bert"))
            pooling = str(bert_cfg.get("pooling", "mean"))
            max_seq_length = int(require(bert_cfg, "max_seq_length", where="features.bert"))
            device = str(require(bert_cfg, "device", where="features.bert"))
            logreg_cfg = get_dict(bert_cfg, "logreg", where="features.bert")

            X_all = bert_embeddings_all(
                X_text,
                model_name=model_name,
                max_seq_length=max_seq_length,
                device=device,
                batch_size=bert_bs,
                pooling=pooling,
            )

            results["methods"]["bert"] = evaluate_with_precomputed_folds(
                X_all,
                y,
                folds,
                logreg_cfg=logreg_cfg,
                average=average,
                zero_division=zero_division,
                print_report=print_report,
                print_cm=print_cm,
                label_names=label_names,
                method_name="bert",
            )

        elif m == "glove":
            glove_cfg = get_dict(feat_cfg, "glove", where="features")
            emb_path = str(require(glove_cfg, "embeddings_path", where="features.glove"))
            emb_dim = int(require(glove_cfg, "embedding_dim", where="features.glove"))
            lowercase = bool(glove_cfg.get("lowercase", False))
            remove_stopwords = bool(glove_cfg.get("remove_stopwords", False))
            stopwords_lang = glove_cfg.get("stopwords_lang", None)
            pooling = str(glove_cfg.get("pooling", "sum_l2norm"))
            logreg_cfg = get_dict(glove_cfg, "logreg", where="features.glove")

            # Chinese-ready knobs (YAML-driven)
            tokenizer = str(glove_cfg.get("tokenizer", "whitespace")).strip().lower()
            max_words = glove_cfg.get("max_words", None)
            max_words_i = None if max_words is None else int(max_words)

            X_all = glove_embeddings_all(
                X_text,
                embeddings_path=emb_path,
                embedding_dim=emb_dim,
                lowercase=lowercase,
                remove_stopwords=remove_stopwords,
                stopwords_lang=None if stopwords_lang is None else str(stopwords_lang),
                pooling=pooling,
                tokenizer=tokenizer,
                max_words=max_words_i,
            )

            results["methods"]["glove"] = evaluate_with_precomputed_folds(
                X_all,
                y,
                folds,
                logreg_cfg=logreg_cfg,
                average=average,
                zero_division=zero_division,
                print_report=print_report,
                print_cm=print_cm,
                label_names=label_names,
                method_name="glove",
            )

        elif m == "gemma":
            gemma_cfg = get_dict(feat_cfg, "gemma", where="features")
            model_name = str(require(gemma_cfg, "model_name", where="features.gemma"))
            pooling = str(gemma_cfg.get("pooling", "mean"))
            max_seq_length = int(require(gemma_cfg, "max_seq_length", where="features.gemma"))
            device = str(require(gemma_cfg, "device", where="features.gemma"))
            logreg_cfg = get_dict(gemma_cfg, "logreg", where="features.gemma")

            X_all = gemma_embeddings_all(
                X_text,
                model_name=model_name,
                max_seq_length=max_seq_length,
                device=device,
                batch_size=gemma_bs,
                pooling=pooling,
            )

            results["methods"]["gemma"] = evaluate_with_precomputed_folds(
                X_all,
                y,
                folds,
                logreg_cfg=logreg_cfg,
                average=average,
                zero_division=zero_division,
                print_report=print_report,
                print_cm=print_cm,
                label_names=label_names,
                method_name="gemma",
            )

        else:
            raise ValueError(f"Unknown method: {method!r} (allowed: tfidf, bert, glove, gemma)")

        if print_method_summary:
            sm = results["methods"][m]["summary"]
            print(f"\n[Summary] {m.upper()}  (mean ± std)")
            print(f"  Accuracy : {sm['accuracy_mean']:.4f} ± {sm['accuracy_std']:.4f}")
            print(f"  Precision: {sm['precision_mean']:.4f} ± {sm['precision_std']:.4f}")
            print(f"  Recall   : {sm['recall_mean']:.4f} ± {sm['recall_std']:.4f}")
            print(f"  F1       : {sm['f1_mean']:.4f} ± {sm['f1_std']:.4f}")

    ensure_parent(metrics_output)
    metrics_output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] Saved CV metrics to: {metrics_output}")
    return str(metrics_output)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper-style K-fold CV evaluator (YAML-driven).")
    p.add_argument("--config", type=str, default=None, help="Path to config_text.yaml")
    p.add_argument("--methods", nargs="+", default=None, help="Override methods: tfidf bert glove gemma")
    p.add_argument("--reuse-folds", action="store_true", help="Reuse existing folds indices JSON.")
    return p

def cli_main() -> None:
    args = build_arg_parser().parse_args()
    run_evaluate_cv(
        config_path=args.config,
        methods_override=None if args.methods is None else [str(x).strip() for x in args.methods],
        reuse_folds=bool(args.reuse_folds),
    )

if __name__ == "__main__":
    cli_main()
