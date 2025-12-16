# -*- coding: utf-8 -*-
"""
cv_features.py

Feature extraction only:
- TF-IDF
- BERT sentence embeddings (mean / CLS / last-N-layers concat + mean)
- GloVe/fastText-style static word embeddings (sentence embedding)
- Gemma sentence embeddings

NOTE (paper-strict):
- YAML often loads ngram_range as a list [a, b]
- Newer scikit-learn requires CountVectorizer.ngram_range to be a tuple (a, b)
  -> We coerce list->tuple in TF-IDF helpers (single responsibility, no config hacks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# =========================
# TF-IDF helpers (strict)
# =========================
def _coerce_ngram_range(v: Any) -> Tuple[int, int]:
    """
    Coerce YAML-loaded ngram_range into sklearn-compatible tuple[int,int].

    Accept:
      - (2,4)
      - [2,4]
      - "2,4" / "(2,4)"
    """
    if v is None:
        return (1, 1)

    if isinstance(v, tuple) and len(v) == 2:
        return (int(v[0]), int(v[1]))

    if isinstance(v, list) and len(v) == 2:
        return (int(v[0]), int(v[1]))

    if isinstance(v, str):
        s = v.strip().replace("(", "").replace(")", "")
        if "," in s:
            a, b = s.split(",", 1)
            return (int(a.strip()), int(b.strip()))

    raise ValueError(f"Invalid ngram_range: {v!r}. Expect [a,b] or (a,b) or 'a,b'.")

def _sanitize_vectorizer_cfg(vectorizer_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make CountVectorizer params sklearn-safe, without mutating caller dict.
    Currently only fixes ngram_range (list->tuple).
    """
    cfg = dict(vectorizer_cfg or {})
    if "ngram_range" in cfg:
        cfg["ngram_range"] = _coerce_ngram_range(cfg["ngram_range"])
    return cfg

def tfidf_features_all(
    X: Sequence[str],
    *,
    vectorizer_cfg: Dict[str, Any],
    transformer_cfg: Dict[str, Any],
) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    vect_cfg = _sanitize_vectorizer_cfg(vectorizer_cfg)
    vect = CountVectorizer(**(vect_cfg or {}))
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

    vect_cfg = _sanitize_vectorizer_cfg(vectorizer_cfg)
    vect = CountVectorizer(**(vect_cfg or {}))
    tfidf = TfidfTransformer(**(transformer_cfg or {}))

    Xtr = tfidf.fit_transform(vect.fit_transform(X_train)).toarray()
    Xte = tfidf.transform(vect.transform(X_test)).toarray()
    return Xtr, Xte

# =========================
# BERT / Gemma helpers
# =========================
def _masked_mean(last_hidden: "np.ndarray", attention_mask: "np.ndarray") -> "np.ndarray":
    # last_hidden: (B,T,H), mask: (B,T)
    mask = attention_mask.astype(np.float32)
    mask = mask[:, :, None]  # (B,T,1)
    summed = (last_hidden * mask).sum(axis=1)
    denom = mask.sum(axis=1)
    denom = np.clip(denom, 1e-6, None)
    return summed / denom

def bert_embeddings_all(
    X: Sequence[str],
    *,
    model_name: str,
    max_seq_length: int,
    device: str,
    batch_size: int,
    embedding_strategy: str = "mean",   # mean | cls | last4_concat_mean
    last_n_layers: int = 4,            # for last4_concat_mean
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:  # noqa: BLE001
        raise ImportError("BERT evaluation requires torch + transformers installed.") from e

    strat = (embedding_strategy or "mean").strip().lower()
    if strat not in ("mean", "cls", "last4_concat_mean"):
        raise ValueError("features.bert.embedding_strategy must be one of: mean | cls | last4_concat_mean")

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()

    dev = torch.device(device)
    mdl.to(dev)

    out_list: List[np.ndarray] = []
    bs = max(1, int(batch_size))
    ln = max(1, int(last_n_layers))

    need_hidden = (strat == "last4_concat_mean")

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
            if need_hidden:
                outputs = mdl(**enc, output_hidden_states=True)
                hs = outputs.hidden_states  # tuple: (emb, l1, ..., lL)
                if hs is None or len(hs) < (ln + 1):
                    raise ValueError("BERT did not return enough hidden_states for last4_concat_mean.")
                last4 = torch.cat(hs[-ln:], dim=-1)  # (B,T,H*ln)
                last_np = last4.detach().cpu().numpy()
            else:
                outputs = mdl(**enc)
                last = outputs.last_hidden_state  # (B,T,H)
                last_np = last.detach().cpu().numpy()

            att = enc.get("attention_mask")
            att_np = att.detach().cpu().numpy() if att is not None else None

            if strat == "cls":
                sent = last_np[:, 0, :]  # (B,H)
            else:
                if att_np is None:
                    sent = last_np.mean(axis=1)
                else:
                    sent = _masked_mean(last_np, att_np)

        out_list.append(sent.astype(np.float32))

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

    pool = (pooling or "mean").strip().lower()
    if pool != "mean":
        raise ValueError("features.gemma.pooling supports only: 'mean' (paper-aligned).")

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
            last = outputs.last_hidden_state  # (B,T,H)
            last_np = last.detach().cpu().numpy()
            att = enc.get("attention_mask")
            att_np = att.detach().cpu().numpy() if att is not None else None

            if att_np is None:
                sent = last_np.mean(axis=1)
            else:
                sent = _masked_mean(last_np, att_np)

        out_list.append(sent.astype(np.float32))

    return np.concatenate(out_list, axis=0)

# =========================
# GloVe helpers
# =========================
def glove_embeddings_all(
    X: Sequence[str],
    *,
    embeddings_path: str,
    embedding_dim: int,
    lowercase: bool,
    remove_stopwords: bool,
    stopwords_lang: Optional[str],
    pooling: str,
    tokenizer: str = "whitespace",          # jieba | char | whitespace
    max_words: Optional[int] = None,        # cap vocab for RAM safety
) -> np.ndarray:
    """
    Sentence embeddings from static word vectors (GloVe/fastText .vec-like).

    Chinese note:
      - Use tokenizer="jieba" (recommended) or tokenizer="char".
      - tokenizer="whitespace" usually yields near-all OOV for Chinese transcripts.
    """
    p = Path(embeddings_path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")

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

    jieba_mod = None
    if tok_mode == "jieba":
        try:
            import jieba as jieba_mod  # type: ignore
        except Exception as e:
            raise ImportError("glove.tokenizer='jieba' requires: pip install jieba") from e

    emb: Dict[str, np.ndarray] = {}
    dim = int(embedding_dim)

    mw = None if max_words is None else max(0, int(max_words))
    loaded = 0

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < (1 + dim):
                continue  # also skips header like: "2000000 300"
            w = parts[0]
            try:
                vec = np.asarray(parts[1 : 1 + dim], dtype=np.float32)
            except Exception:
                continue
            if vec.shape[0] == dim:
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
            toks = [t.strip() for t in jieba_mod.lcut(s) if t.strip()]  # type: ignore[union-attr]
        elif tok_mode == "char":
            toks = [ch for ch in s if not ch.isspace()]
        else:
            toks = s.split()

        if remove_stopwords and stop:
            toks = [t for t in toks if t not in stop]
        return toks

    pool = (pooling or "sum_l2norm").strip().lower()
    if pool != "sum_l2norm":
        raise ValueError("features.glove.pooling must be 'sum_l2norm' (paper-aligned).")

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
        denom = float(np.sqrt(np.sum(summed * summed))) or 1.0
        return (summed / denom).astype(np.float32)

    return np.stack([_sent_vec(s) for s in X], axis=0)
