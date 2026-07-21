#!/usr/bin/env python3
"""Fully offline RAG: TF-IDF vectors over WHO passages — no network, no Ollama."""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

LANG_ALIASES = {
    "english": "en",
    "yoruba": "yo",
    "yorùbá": "yo",
    "hausa": "ha",
    "igbo": "ig",
    "swahili": "sw",
}


def tokenize(text: str) -> list[str]:
    """NFKC-normalize; keep letters, combining marks, and digits as word tokens."""
    norm = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    buf: list[str] = []
    for ch in norm:
        cat = unicodedata.category(ch)
        if cat[0] in {"L", "M", "N"}:
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def _lang_code(lang: str) -> str:
    key = lang.strip().lower()
    return LANG_ALIASES.get(key, key)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def build_tfidf(texts: list[str]) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Fit a sparse-style TF-IDF matrix; returns vocab, idf vector, and L2-normalized rows."""
    docs = [tokenize(t) for t in texts]
    df: dict[str, int] = {}
    for tokens in docs:
        for tok in set(tokens):
            df[tok] = df.get(tok, 0) + 1

    vocab = {tok: i for i, tok in enumerate(sorted(df))}
    n_docs = max(len(docs), 1)
    idf = np.zeros(len(vocab), dtype=np.float32)
    for tok, i in vocab.items():
        idf[i] = math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0

    mat = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for row, tokens in enumerate(docs):
        if not tokens:
            continue
        counts: dict[int, int] = {}
        for tok in tokens:
            idx = vocab.get(tok)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
        length = float(len(tokens))
        for idx, cnt in counts.items():
            mat[row, idx] = (cnt / length) * idf[idx]

    return vocab, idf, _l2_normalize(mat)


def embed_query(text: str, vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    """Project a query into the fitted TF-IDF space and L2-normalize."""
    tokens = tokenize(text)
    vec = np.zeros(len(vocab), dtype=np.float32)
    if not tokens or not vocab:
        return vec
    counts: dict[int, int] = {}
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    length = float(len(tokens))
    for idx, cnt in counts.items():
        vec[idx] = (cnt / length) * idf[idx]
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def _cosine_rows(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ query


class Retriever:
    """In-memory cosine retriever over corpus passages {id, source, lang, topic, text}."""

    def __init__(
        self,
        passages: list[dict[str, Any]],
        vectors: np.ndarray,
        vocab: dict[str, int],
        idf: np.ndarray,
    ):
        self.passages = passages
        self.vectors = vectors
        self.vocab = vocab
        self.idf = idf

    @classmethod
    def from_passages(cls, passages: list[dict[str, Any]]) -> "Retriever":
        """Fit TF-IDF over passage texts (fully offline)."""
        vocab, idf, vectors = build_tfidf([p["text"] for p in passages])
        return cls(passages, vectors, vocab, idf)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Retriever":
        """Load passages from JSONL; reuse precomputed index beside corpus when present."""
        path = Path(path)
        passages = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        index_dir = path.parent.parent / "index"
        if (index_dir / "tfidf.npz").is_file() and (index_dir / "vocab.json").is_file():
            try:
                return cls.load_index(index_dir, passages)
            except (OSError, ValueError, KeyError):
                pass
        return cls.from_passages(passages)

    def save_index(self, index_dir: str | Path) -> None:
        """Persist TF-IDF artifacts for fast cold-start without re-fitting."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_dir / "tfidf.npz", vectors=self.vectors, idf=self.idf)
        meta = {
            "passage_ids": [p["id"] for p in self.passages],
            "vocab": self.vocab,
        }
        (index_dir / "vocab.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )

    @classmethod
    def load_index(cls, index_dir: str | Path, passages: list[dict[str, Any]]) -> "Retriever":
        """Load a saved index; passage order must match saved passage_ids."""
        index_dir = Path(index_dir)
        data = np.load(index_dir / "tfidf.npz")  # float arrays only — no pickle
        meta = json.loads((index_dir / "vocab.json").read_text(encoding="utf-8"))
        vocab = meta["vocab"]
        saved_ids = [str(x) for x in meta["passage_ids"]]
        by_id = {p["id"]: p for p in passages}
        ordered = [by_id[i] for i in saved_ids if i in by_id]
        if len(ordered) != len(saved_ids):
            raise ValueError("Index passage_ids do not match corpus")
        return cls(ordered, data["vectors"].astype(np.float32), vocab, data["idf"].astype(np.float32))

    def retrieve(
        self,
        query: str,
        lang: str = "English",
        k: int = 4,
        topics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k language-matched passages; optional topic filter before TF-IDF rank."""
        q = embed_query(query, self.vocab, self.idf)
        target = _lang_code(lang)
        topic_set = set(topics) if topics else None

        indices = [
            i
            for i, p in enumerate(self.passages)
            if _lang_code(str(p.get("lang", "en"))) == target
            and (topic_set is None or str(p.get("topic", "")) in topic_set)
        ]
        # Fall back: same language without topic filter, then any language.
        if not indices and topic_set is not None:
            indices = [
                i
                for i, p in enumerate(self.passages)
                if _lang_code(str(p.get("lang", "en"))) == target
            ]
        if not indices:
            indices = list(range(len(self.passages)))

        sub = self.vectors[indices]
        scores = _cosine_rows(q, sub)
        order = np.argsort(-scores)[:k]
        return [self.passages[indices[int(j)]] for j in order]

    def retrieve_by_topics(
        self,
        topics: list[str],
        lang: str = "English",
        k: int = 4,
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Cover requested topics in order, then fill remaining slots by TF-IDF rank."""
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for topic in topics:
            for hit in self.retrieve(query or topic, lang=lang, k=k, topics=[topic]):
                if hit["id"] in seen:
                    continue
                selected.append(hit)
                seen.add(hit["id"])
                if len(selected) >= k:
                    return selected[:k]
        if len(selected) < k:
            for hit in self.retrieve(query or " ".join(topics), lang=lang, k=k):
                if hit["id"] in seen:
                    continue
                selected.append(hit)
                seen.add(hit["id"])
                if len(selected) >= k:
                    break
        return selected[:k]

    @staticmethod
    def as_context(passages: list[dict[str, Any]]) -> str:
        """Render retrieved passages as id-tagged reference lines for the prompt."""
        return "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
