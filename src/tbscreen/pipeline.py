#!/usr/bin/env python3
"""Full pipeline: chest X-ray → vision screening → RAG retrieval → LLM clinical interpretation."""

from __future__ import annotations

from typing import Any

from llama_cpp import Llama

from . import core, llm
from .rag import Retriever


def _retrieval_query(vision_result: dict[str, Any]) -> str:
    """Build a retrieval query from the vision model's output."""
    risk, triage = core.classify_risk(vision_result.get("tb_probability", 0.0))
    parts = [
        "tuberculosis",
        f"{risk} risk",
        f"triage {triage}",
        vision_result.get("screening_result", ""),
        vision_result.get("dominant_zone", ""),
    ]
    return " ".join(p for p in parts if p)


def screen_and_interpret(
    model: Llama,
    retriever: Retriever,
    vision_result: dict[str, Any],
    lang: str = "English",
    k: int = 4,
) -> dict[str, Any]:
    """Full turn: vision result → retrieve WHO guidelines → grounded clinical interpretation."""
    query = _retrieval_query(vision_result)
    hits = retriever.retrieve(query, lang=lang, k=k)
    context = Retriever.as_context(hits)

    interpretation = llm.interpret(
        model,
        vision_result,
        context,
        lang=lang,
        valid_sources=[h["id"] for h in hits],
    )

    return {
        "vision_result": vision_result,
        "risk_level": interpretation["risk_level"] if interpretation else "unknown",
        "triage": interpretation["triage"] if interpretation else "unknown",
        "interpretation": interpretation,
        "retrieved_sources": [h["id"] for h in hits],
    }


def answer_clinical_question(
    model: Llama,
    retriever: Retriever,
    question: str,
    lang: str = "English",
    k: int = 4,
) -> dict[str, Any]:
    """Text clinical Q&A path used by metadata test prompts and the /ask UI."""
    hits = retriever.retrieve(question, lang=lang, k=k)
    context = Retriever.as_context(hits)
    answer = llm.answer_question(
        model,
        question,
        context,
        lang=lang,
        valid_sources=[h["id"] for h in hits],
    )
    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": [h["id"] for h in hits],
    }
