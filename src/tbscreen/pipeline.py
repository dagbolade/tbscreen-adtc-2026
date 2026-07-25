"""Full pipeline: chest X-ray → vision screening → topic RAG → LLM clinical interpretation."""

from __future__ import annotations

from typing import Any

from llama_cpp import Llama

from . import core, llm
from .rag import Retriever


def _patient_kwargs(patient_context: dict[str, Any] | None) -> dict[str, Any]:
    if not patient_context:
        return {}
    keys = ("age_years", "cough_weeks", "has_tb_symptoms", "hiv_positive", "household_contact")
    return {k: patient_context[k] for k in keys if k in patient_context}


def screen_and_interpret(
    model: Llama,
    retriever: Retriever,
    vision_result: dict[str, Any],
    lang: str = "English",
    k: int = 4,
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full turn: vision → WHO-aligned triage → topic-localized RAG → grounded interpretation."""
    decision = core.decide_triage(
        vision_result.get("tb_probability", 0.0),
        **_patient_kwargs(patient_context),
    )
    topics = core.topics_for_decision(decision)
    # English topic labels rank within the language-filtered / topic-filtered subset.
    query = " ".join(topics) + " tuberculosis screening triage"
    hits = retriever.retrieve_by_topics(topics, lang=lang, k=k, query=query)
    context = Retriever.as_context(hits)

    interpretation = llm.interpret(
        model,
        vision_result,
        context,
        lang=lang,
        valid_sources=[h["id"] for h in hits],
        decision=decision,
    )

    return {
        "vision_result": vision_result,
        "decision": decision,
        "risk_level": decision["risk_level"],
        "triage": decision["triage"],
        "interpretation": interpretation,
        "retrieved_sources": [h["id"] for h in hits],
        "retrieval_topics": topics,
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
