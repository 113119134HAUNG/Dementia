# -*- coding: utf-8 -*-
"""
cv_features.py

Feature extraction only:
- TF-IDF
- BERT sentence embeddings
- GloVe/fastText-style static word embeddings (sentence embedding)
- Gemma sentence embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================
# TF-IDF
# ============================================================
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

# ============================================================
# Helpers: mask-aware mean pooling
# ============================================================
def _masked_mean_pool(last_hidden_state, attention_mask):
    """
    last_hidden_state: (B, T, H)
    attention_mask:    (B, T) 1 for tokens, 0 for padding
    """
    # (B, T, 1)
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)  # (B, H)
    denom = mask.sum(dim=1).clamp(min=1.0)          # (B, 1)
    return summed / denom

# ============================================================
# BERT sentence embeddings
# ============================================================
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

    pooling = (pooling or "mean").strip().lower()
    if pooling != "mean":
        raise ValueError("features.bert.pooling supports only: 'mean' (paper-aligned).")

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
            attn = enc.get("attention_mask", None)
            if attn is None:
                # fallback (should not happen for HF tokenizers)
                sent = last.mean(dim=1)
            else:
                sent = _masked_mean_pool(last, attn)

        out_list.append(sent.detach().cpu().numpy())

    return np.concatenate(out_list, axis=0)

# ============================================================
# Gemma sentence embeddings
# ============================================================
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
    # Some decoder-only tokenizers have no pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    dev = torch.device(device)
    mdl.to(dev)

    out_list: List[np.ndarray] = []
    bs = max(1, int(batch_size))

    pooling = (pooling or "mean").strip().lower()
    if pooling != "mean":
        raise ValueError("features.gemma.pooling supports only: 'mean' (paper-aligned).")

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
            attn = enc.get("attention_mask", None)
            if attn is None:
                sent = last.mean(dim=1)
            else:
                sent = _masked_mean_pool(last, attn)

        out_list.append(sent.detach().cpu().numpy())

    return np.concatenate(out_list, axis=0)

# ============================================================
# Static word vectors sentence embeddings (GloVe / fastText .vec)
# ============================================================
def glove_embeddings_all(
    X: Sequence[str],
    *,
    embeddings_path: str,
    embedding_dim: int,
    lowercase: bool,
    remove_stopwords: bool,
    stopwords_lang: Optional[str],
    pooling: str,
    tokenizer: str = "whitespace",          # "jieba" | "char" | "whitespace"
    max_words: Optional[int] = None,        # cap vocab for RAM safety
) -> np.ndarray:
    """
    Sentence embeddings from static word vectors (GloVe/fastText .vec-like).

    Notes
    -----
    - Supports fastText .vec header: first line "N D" will be skipped automatically.
    - For Chinese, prefer tokenizer="jieba" (recommended) or tokenizer="char".
      tokenizer="whitespace" usually yields near-all OOV for Chinese transcripts.
    - pooling (paper-aligned): only "sum_l2norm"
    """
    p = Path(embeddings_path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")

    pooling = (pooling or "sum_l2norm").strip().lower()
    if pooling != "sum_l2norm":
        raise ValueError("features.glove.pooling must be 'sum_l2norm' (paper-aligned).")

    dim = int(embedding_dim)
    if dim <= 0:
        raise ValueError("features.glove.embedding_dim must be > 0.")

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

    tok_mode = (tokenizer or "whitespace").strip().lower()
    if tok_mode not in ("whitespace", "jieba", "char"):
        raise ValueError("features.glove.tokenizer must be one of: 'whitespace', 'jieba', 'char'.")

    # Cap vocab size to avoid RAM explosion
    mw = None if max_words is None else max(0, int(max_words))
    if mw == 0:
        # explicit: load none
        emb: Dict[str, np.ndarray] = {}
    else:
        emb = {}
        loaded = 0

        def _is_fasttext_header(parts: List[str]) -> bool:
            # fastText .vec often starts with: "<n_words> <dim>"
            if len(parts) != 2:
                return False
            return parts[0].isdigit() and parts[1].isdigit()

        with p.open("r", encoding="utf-8", errors="ignore") as f:
            first = True
            for line in f:
                parts = line.rstrip().split()
                if not parts:
                    continue

                # Skip header line if present
                if first:
                    first = False
                    if _is_fasttext_header(parts):
                        continue

                if len(parts) < (1 + dim):
                    continue

                w = parts[0]
                try:
                    vec = np.asarray(parts[1 : 1 + dim], dtype=np.float32)
                except Exception:
                    continue

                if vec.shape[0] != dim:
                    continue

                emb[w] = vec
                loaded += 1
                if mw is not None and loaded >= mw:
                    break

    def _tokenize(s: str) -> List[str]:
        s = (s or "").strip()
        if not s:
            return []
        if lowercase:
            s = s.lower()

        if tok_mode == "jieba":
            try:
                import jieba  # type: ignore
            except Exception as e:
                raise ImportError("glove.tokenizer='jieba' requires: pip install jieba") from e
            toks = [t.strip() for t in jieba.lcut(s) if t.strip()]
        elif tok_mode == "char":
            toks = [ch for ch in s if not ch.isspace()]
        else:
            toks = s.split()

        if remove_stopwords and stop:
            toks = [t for t in toks if t not in stop]
        return toks

    def _sent_vec(s: str) -> np.ndarray:
        toks = _tokenize(s)
        if not toks:
            return np.zeros(dim, dtype=np.float32)

        vecs: List[np.ndarray] = []
        for t in toks:
            v = emb.get(t)
            if v is not None:
                vecs.append(v)

        if not vecs:
            return np.zeros(dim, dtype=np.float32)

        mat = np.stack(vecs, axis=0)
        summed = mat.sum(axis=0)

        denom = float(np.sqrt(np.sum(summed * summed))) or 1.0
        return (summed / denom).astype(np.float32)

    return np.stack([_sent_vec(s) for s in X], axis=0)
