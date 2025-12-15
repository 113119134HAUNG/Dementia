# -*- coding: utf-8 -*-
"""
cv_features.py

Feature extraction only:
- TF-IDF
- BERT sentence embeddings
- GloVe sentence embeddings
- Gemma sentence embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def tfidf_features_all(
    X: Sequence[str],
    *,
    vectorizer_cfg: Dict[str, Any],
    transformer_cfg: Dict[str, Any],
) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    vect = CountVectorizer(**(vectorizer_cfg or {}))
    tfidf = TfidfTransformer(**(transformer_cfg or {}))

    counts = vect.fit_transform(X)
    mat = tfidf.fit_transform(counts)
    return mat.toarray()


def tfidf_features_fold_fit(
    X_train: Sequence[str],
    X_test: Sequence[str],
    *,
    vectorizer_cfg: Dict[str, Any],
    transformer_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    vect = CountVectorizer(**(vectorizer_cfg or {}))
    tfidf = TfidfTransformer(**(transformer_cfg or {}))

    Xtr = tfidf.fit_transform(vect.fit_transform(X_train)).toarray()
    Xte = tfidf.transform(vect.transform(X_test)).toarray()
    return Xtr, Xte


def bert_embeddings_all(
    X: Sequence[str],
    *,
    model_name: str,
    max_seq_length: int,
    device: str,
    batch_size: int,
    pooling: str = "mean",
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:  # noqa: BLE001
        raise ImportError("BERT evaluation requires torch + transformers installed.") from e

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    dev = torch.device(device)
    mdl.to(dev)

    out_list: List[np.ndarray] = []
    bs = max(1, int(batch_size))

    for i in range(0, len(X), bs):
        batch = X[i : i + bs]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_seq_length),
            return_tensors="pt",
            add_special_tokens=True,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            outputs = mdl(**enc)
            last = outputs.last_hidden_state  # (B, T, H)

            if pooling == "mean":
                sent = last.mean(dim=1)
            else:
                raise ValueError("features.bert.pooling supports only: 'mean' (paper-aligned).")

        out_list.append(sent.detach().cpu().numpy())

    return np.concatenate(out_list, axis=0)


def gemma_embeddings_all(
    X: Sequence[str],
    *,
    model_name: str,
    max_seq_length: int,
    device: str,
    batch_size: int,
    pooling: str = "mean",
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:  # noqa: BLE001
        raise ImportError("Gemma evaluation requires torch + transformers installed.") from e

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    dev = torch.device(device)
    mdl.to(dev)

    out_list: List[np.ndarray] = []
    bs = max(1, int(batch_size))

    for i in range(0, len(X), bs):
        batch = X[i : i + bs]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_seq_length),
            return_tensors="pt",
            add_special_tokens=True,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            outputs = mdl(**enc)
            last = outputs.last_hidden_state  # (B, T, H)

            if pooling == "mean":
                sent = last.mean(dim=1)
            else:
                raise ValueError("features.gemma.pooling supports only: 'mean' (paper-aligned).")

        out_list.append(sent.detach().cpu().numpy())

    return np.concatenate(out_list, axis=0)


def glove_embeddings_all(
    X: Sequence[str],
    *,
    embeddings_path: str,
    embedding_dim: int,
    lowercase: bool,
    remove_stopwords: bool,
    stopwords_lang: Optional[str],
    pooling: str,
) -> np.ndarray:
    p = Path(embeddings_path)
    if not p.exists():
        raise FileNotFoundError(f"GloVe embeddings file not found: {p}")

    # stopwords (optional; no downloads)
    stop: set = set()
    if remove_stopwords and stopwords_lang and str(stopwords_lang).lower() == "english":
        try:
            from nltk.corpus import stopwords as _sw  # type: ignore
            stop = set(_sw.words("english"))
        except Exception:
            stop = {
                "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
                "is", "are", "was", "were", "be", "been", "being",
            }

    emb: Dict[str, np.ndarray] = {}
    dim = int(embedding_dim)

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < (1 + dim):
                continue
            w = parts[0]
            try:
                vec = np.asarray(parts[1 : 1 + dim], dtype=np.float32)
            except Exception:
                continue
            if vec.shape[0] == dim:
                emb[w] = vec

    def _tokenize(s: str) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        if lowercase:
            s = s.lower()
        toks = s.split()
        if remove_stopwords and stop:
            toks = [t for t in toks if t not in stop]
        return toks

    def _sent_vec(s: str) -> np.ndarray:
        toks = _tokenize(s)
        vecs: List[np.ndarray] = []
        for t in toks:
            v = emb.get(t)
            if v is not None:
                vecs.append(v)

        if not vecs:
            return np.zeros(dim, dtype=np.float32)

        mat = np.stack(vecs, axis=0)
        summed = mat.sum(axis=0)

        if pooling == "sum_l2norm":
            denom = float(np.sqrt(np.sum(summed * summed))) or 1.0
            return (summed / denom).astype(np.float32)

        raise ValueError("features.glove.pooling must be 'sum_l2norm' (paper-aligned).")

    return np.stack([_sent_vec(s) for s in X], axis=0)
