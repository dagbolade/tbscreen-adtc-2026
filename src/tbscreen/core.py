#!/usr/bin/env python3
"""Clinical schema, prompts, YAML repair, and anti-hallucination for TB screening."""

from __future__ import annotations

import re
from typing import Any, Iterable

import yaml

# --- Clinical risk levels mapped to vision model probability ---

RISK_THRESHOLDS = {
    "high": 0.65,      # above screening threshold
    "moderate": 0.40,  # concerning but below threshold
    "low": 0.0,        # low probability
}

TRIAGE_MAP = {
    "high": "refer",      # refer for confirmatory sputum test
    "moderate": "retest",  # retest or monitor
    "low": "clear",        # no immediate TB concern
}

VALID_TRIAGE = {"refer", "retest", "clear"}
VALID_RISK = {"high", "moderate", "low"}

# End markers across model families so a leaked token never breaks parsing.
STOP_TOKENS = ["<end_of_turn>", "<|im_end|>", "<turn|>", "<|eot_id|>", "<|end_of_text|>", "<eos>"]


def classify_risk(tb_probability: float) -> tuple[str, str]:
    """Map TB probability to (risk_level, triage) using calibrated thresholds."""
    if tb_probability >= RISK_THRESHOLDS["high"]:
        return "high", "refer"
    elif tb_probability >= RISK_THRESHOLDS["moderate"]:
        return "moderate", "retest"
    return "low", "clear"


# --- Prompt templates ---

INTERPRET_INSTRUCTION = """You are a clinical decision-support assistant for TB screening in resource-limited settings. A chest X-ray has been screened by an AI model. Using ONLY the reference passages below, produce a structured clinical interpretation.

CRITICAL RULES:
- This is decision SUPPORT, not diagnosis. Always include a disclaimer.
- NEVER state clinical facts not supported by the reference passages.
- Every clinical claim must cite a source id from the references.
- If zone activations are provided, briefly note which lung region contributed most to the score. Do not invent precise anatomical findings beyond the zone labels.

Reply in {lang}. Output YAML only. No markdown fences, no prose, no explanations.
Top-level keys must be exactly: interpretation, recommendation, education, cautions, sources.
- interpretation: 2-3 sentences explaining what the screening result means for this patient.
- recommendation: specific next steps for the health worker (what to do now).
- education: a list of 2-3 simple points to explain to the patient about TB.
- cautions: a list of safety warnings (always include "this is a screening tool, not a diagnosis").
- sources: a list of reference passage ids that support the interpretation.

Format exactly like this (each field on its own line):
interpretation: <2-3 sentences>
recommendation: <next steps>
education:
  - <point 1>
  - <point 2>
cautions:
  - <warning 1>
  - <warning 2>
sources:
  - <passage id>

Reference passages:
{context}

Screening result:
{screening_summary}"""


def build_screening_summary(vision_result: dict[str, Any]) -> str:
    """Format vision model output into a text summary for the LLM prompt."""
    prob = vision_result.get("tb_probability", 0.0)
    result = vision_result.get("screening_result", "UNKNOWN")
    risk, triage = classify_risk(prob)
    lines = [
        f"TB probability: {prob:.1%}",
        f"Screening result: {result}",
        f"Risk level: {risk}",
        f"Recommended triage: {triage}",
    ]
    zones = vision_result.get("zone_activations")
    if isinstance(zones, dict) and zones:
        zone_str = ", ".join(f"{k}={v:.0%}" for k, v in zones.items())
        lines.append(f"Zone activations (occlusion sensitivity): {zone_str}")
        dominant = vision_result.get("dominant_zone")
        if dominant:
            lines.append(f"Dominant zone: {dominant}")
    return "\n".join(lines)


def interpret_user_content(
    screening_summary: str,
    context: str,
    lang: str = "English",
) -> str:
    """Build the full user prompt for clinical interpretation."""
    return INTERPRET_INSTRUCTION.format(
        lang=lang,
        context=context,
        screening_summary=screening_summary,
    )


QA_INSTRUCTION = """You are a clinical decision-support assistant for TB care in resource-limited settings. Answer the health worker's question using ONLY the reference passages below.

CRITICAL RULES:
- This is decision SUPPORT, not diagnosis. Always include a disclaimer.
- NEVER state clinical facts not supported by the reference passages.
- Every clinical claim must cite a source id from the references.

Reply in {lang}. Output YAML only. No markdown fences, no prose, no explanations.
Top-level keys must be exactly: answer, recommendation, education, cautions, sources.
- answer: 2-4 sentences answering the question for a community health worker.
- recommendation: specific next steps the health worker can take now.
- education: a list of 2-3 simple points suitable to share with a patient or caregiver.
- cautions: a list of safety warnings (always include that this is decision support, not a diagnosis).
- sources: a list of reference passage ids that support the answer.

Format exactly like this (each field on its own line):
answer: <2-4 sentences>
recommendation: <next steps>
education:
  - <point 1>
  - <point 2>
cautions:
  - <warning 1>
  - <warning 2>
sources:
  - <passage id>

Reference passages:
{context}

Question:
{question}"""


def qa_user_content(question: str, context: str, lang: str = "English") -> str:
    """Build the full user prompt for grounded clinical Q&A."""
    return QA_INSTRUCTION.format(lang=lang, context=context, question=question)


def sanitize_qa(
    obj: Any,
    valid_sources: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Sanitize clinical Q&A YAML; drop fabricated source ids."""
    if not isinstance(obj, dict):
        return None

    allowed = set(valid_sources) if valid_sources is not None else None
    raw_sources = _str_list(obj.get("sources"))
    sources = [s for s in raw_sources if s in allowed] if allowed is not None else raw_sources

    cautions = _str_list(obj.get("cautions"))
    has_disclaimer = any("screening" in c.lower() or "diagnosis" in c.lower() or "decision support" in c.lower() for c in cautions)
    if not has_disclaimer:
        cautions.append(
            "This is AI decision support, not a definitive diagnosis. Confirmatory testing and clinician judgment are required."
        )

    return {
        "answer": str(obj.get("answer", "")).strip(),
        "recommendation": str(obj.get("recommendation", "")).strip(),
        "education": _str_list(obj.get("education")),
        "cautions": cautions,
        "sources": sources,
    }


# --- YAML cleaning and parsing (proven infrastructure from prior work) ---

def clean_yaml_text(raw: str) -> str:
    """Strip markdown fences, stop tokens, and preamble from model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("yaml"):
            text = text[4:].lstrip()
        idx = text.rfind("```")
        if idx != -1:
            text = text[:idx].strip()

    cut = len(text)
    for tok in STOP_TOKENS:
        i = text.find(tok)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def parse_yaml(raw: str) -> Any:
    """Clean then YAML-parse model output; None on failure."""
    try:
        return yaml.safe_load(clean_yaml_text(raw))
    except yaml.YAMLError:
        return None


def dump_yaml(obj: Any) -> str:
    """Dump to YAML preserving African-language characters."""
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).strip()


# --- Schema sanitizers with anti-hallucination ---

def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" or text.lower() == "null" else text


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def sanitize_interpretation(
    obj: Any,
    risk_level: str,
    triage: str,
    valid_sources: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Sanitize and validate LLM clinical output; drop fabricated sources."""
    if not isinstance(obj, dict):
        return None

    allowed = set(valid_sources) if valid_sources is not None else None

    # Filter sources to only those from retrieved passages
    raw_sources = _str_list(obj.get("sources"))
    if allowed is not None:
        sources = [s for s in raw_sources if s in allowed]
    else:
        sources = raw_sources

    # Ensure cautions always includes the screening disclaimer
    cautions = _str_list(obj.get("cautions"))
    has_disclaimer = any("screening" in c.lower() or "diagnosis" in c.lower() for c in cautions)
    if not has_disclaimer:
        cautions.append("This is an AI screening tool, not a definitive diagnosis. Confirmatory testing is required.")

    return {
        "risk_level": risk_level,
        "triage": triage,
        "interpretation": str(obj.get("interpretation", "")).strip(),
        "recommendation": str(obj.get("recommendation", "")).strip(),
        "education": _str_list(obj.get("education")),
        "cautions": cautions,
        "sources": sources,
    }
