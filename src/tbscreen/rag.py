#!/usr/bin/env python3
"""Offline RAG: embed a clinical corpus and retrieve grounded passages by cosine similarity."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
import urllib.request

# Multilingual sentence embedder served by Ollama.
EMBED_MODEL = os.environ.get("TBSCREEN_EMBED", "paraphrase-multilingual")
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"


def embed(text: str) -> list[float]:
    """Get embeddings from Ollama's local embedding API."""
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(OLLAMA_EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    """In-memory cosine retriever over corpus passages {id, source, lang, text}."""

    def __init__(self, passages: list[dict[str, Any]], vectors: list[list[float]]):
        self.passages = passages
        self.vectors = vectors

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Retriever":
        """Load passages from a JSONL file and embed them at init."""
        passages = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        vectors = [embed(p["text"]) for p in passages]
        return cls(passages, vectors)

    def retrieve(self, query: str, lang: str = "English", k: int = 4) -> list[dict[str, Any]]:
        """Return the top-k passages matching the language, most similar to the query."""
        q = embed(query)
        
        filtered_pvs = []
        for p, v in zip(self.passages, self.vectors):
            p_lang = p.get("lang", "en").lower()
            target_lang = lang.lower()
            
            lang_mapping = {
                "english": "en",
                "yoruba": "yo",
                "hausa": "ha",
                "igbo": "ig",
                "swahili": "sw"
            }
            target_code = lang_mapping.get(target_lang, target_lang)
            p_code = lang_mapping.get(p_lang, p_lang)
            
            if p_code == target_code:
                filtered_pvs.append((p, v))
                
        if not filtered_pvs:
            filtered_pvs = list(zip(self.passages, self.vectors))
            
        scored = sorted(filtered_pvs, key=lambda pv: _cosine(q, pv[1]), reverse=True)
        return [p for p, _ in scored[:k]]

    @staticmethod
    def as_context(passages: list[dict[str, Any]]) -> str:
        """Render retrieved passages as id-tagged reference lines for the prompt."""
        return "\n".join(f"[{p['id']}] {p['text']}" for p in passages)
