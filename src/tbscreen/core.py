#!/usr/bin/env python3
"""Clinical schema, WHO-aligned triage policy, prompts, and anti-hallucination sanitizers."""

from __future__ import annotations

from typing import Any, Iterable

import yaml

# Vision probability bands (screen status only — not a clearance decision).
RISK_THRESHOLDS = {
    "high": 0.65,
    "moderate": 0.40,
    "low": 0.0,
}

# WHO CAD guidance commonly evaluated for ages ≥15; pediatric CXR CAD is out of scope here.
CAD_MIN_AGE_YEARS = 15

VALID_TRIAGE = {"refer", "retest", "symptom_followup", "monitor"}
VALID_RISK = {"high", "moderate", "low"}
VALID_SCREEN = {"POSITIVE", "NEGATIVE", "INDETERMINATE", "OUT_OF_SCOPE"}

STOP_TOKENS = [
    "<end_of_turn>",
    "<|im_end|>",
    "<turn|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<eos>",
]


def screen_risk(tb_probability: float) -> str:
    """Map CAD probability to a screen risk band (not a clinical clearance)."""
    if tb_probability >= RISK_THRESHOLDS["high"]:
        return "high"
    if tb_probability >= RISK_THRESHOLDS["moderate"]:
        return "moderate"
    return "low"


def classify_risk(tb_probability: float) -> tuple[str, str]:
    """Backward-compatible (risk, triage) from probability alone; prefer decide_triage()."""
    risk = screen_risk(tb_probability)
    legacy = {"high": "refer", "moderate": "retest", "low": "monitor"}
    return risk, legacy[risk]


def decide_triage(
    tb_probability: float,
    *,
    age_years: float | None = None,
    cough_weeks: float | None = None,
    has_tb_symptoms: bool | None = None,
    hiv_positive: bool | None = None,
    household_contact: bool | None = None,
) -> dict[str, Any]:
    """WHO-aligned decision support: CAD screen + age/symptoms/risk factors → triage action."""
    risk = screen_risk(tb_probability)
    screen_positive = tb_probability >= RISK_THRESHOLDS["high"]
    symptomatic = bool(has_tb_symptoms) or (cough_weeks is not None and cough_weeks >= 2)
    high_risk_group = bool(hiv_positive) or bool(household_contact)

    out_of_scope = age_years is not None and age_years < CAD_MIN_AGE_YEARS
    reasons: list[str] = []

    if out_of_scope:
        triage = "refer"
        screen_status = "OUT_OF_SCOPE"
        reasons.append(
            f"CAD CXR screening in this workflow is intended for ages ≥{CAD_MIN_AGE_YEARS}; "
            "refer for age-appropriate clinical evaluation."
        )
    elif screen_positive:
        triage = "refer"
        screen_status = "POSITIVE"
        reasons.append("High CAD probability — refer for confirmatory bacteriological testing.")
    elif risk == "moderate":
        triage = "retest"
        screen_status = "INDETERMINATE"
        reasons.append("Borderline CAD score — retest or proceed to sputum testing if symptomatic.")
        if symptomatic or high_risk_group:
            triage = "refer"
            reasons.append("Symptoms or high-risk group with borderline screen → prioritize confirmatory testing.")
    else:
        screen_status = "NEGATIVE"
        if symptomatic or high_risk_group:
            # Negative CXR does not clear a symptomatic or high-risk patient.
            triage = "symptom_followup"
            reasons.append(
                "Negative CAD screen does not rule out TB when symptoms or risk factors are present; "
                "continue clinical evaluation and consider sputum testing."
            )
        else:
            triage = "monitor"
            reasons.append(
                "Low CAD probability in an asymptomatic patient without named risk factors — "
                "advise return if symptoms develop."
            )

    return {
        "risk_level": risk,
        "triage": triage,
        "screen_status": screen_status,
        "cad_in_scope": not out_of_scope,
        "policy_reasons": reasons,
        "patient_context": {
            "age_years": age_years,
            "cough_weeks": cough_weeks,
            "has_tb_symptoms": has_tb_symptoms,
            "hiv_positive": hiv_positive,
            "household_contact": household_contact,
            "symptomatic": symptomatic,
            "high_risk_group": high_risk_group,
        },
    }


def topics_for_decision(decision: dict[str, Any]) -> list[str]:
    """Language-independent topic ids for RAG based on triage/screen status."""
    triage = decision.get("triage")
    topics = ["screening", "patient_education"]
    if triage == "refer":
        topics.append("triage_refer")
    elif triage == "retest":
        topics.append("triage_retest")
    elif triage in {"symptom_followup", "monitor"}:
        topics.append("triage_clear")
        if decision.get("patient_context", {}).get("symptomatic"):
            topics.append("symptoms")
    if decision.get("patient_context", {}).get("hiv_positive"):
        topics.append("hiv")
    if not decision.get("cad_in_scope", True):
        topics.append("children")
    return topics


INTERPRET_INSTRUCTION = """You are a clinical decision-support assistant for TB screening in resource-limited settings. A chest X-ray has been screened by an AI model. Using ONLY the reference passages below, produce a structured clinical interpretation.

CRITICAL RULES:
- This is decision SUPPORT, not diagnosis. Always include a disclaimer.
- NEVER state clinical facts not supported by the reference passages.
- Every clinical claim must cite a source id from the references.
- A negative CAD screen does NOT clear a symptomatic patient.
- If zone activations are provided, treat them as unvalidated image-quadrant occlusion sensitivity, not anatomy or a heatmap. Do not invent precise anatomical findings.
- If the workflow marks CAD out of scope (e.g. age < 15), say so and recommend age-appropriate evaluation.

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


def build_screening_summary(
    vision_result: dict[str, Any],
    decision: dict[str, Any] | None = None,
) -> str:
    """Format vision + policy decision into a text summary for the LLM prompt."""
    prob = vision_result.get("tb_probability", 0.0)
    decision = decision or decide_triage(prob)
    lines = [
        f"TB probability: {prob:.1%}",
        f"Vision screening label: {vision_result.get('screening_result', 'UNKNOWN')}",
        f"Policy screen status: {decision.get('screen_status')}",
        f"Risk level: {decision.get('risk_level')}",
        f"Recommended triage: {decision.get('triage')}",
        f"CAD in scope: {decision.get('cad_in_scope')}",
    ]
    ctx = decision.get("patient_context") or {}
    for key in ("age_years", "cough_weeks", "has_tb_symptoms", "hiv_positive", "household_contact"):
        if ctx.get(key) is not None:
            lines.append(f"{key}: {ctx[key]}")
    for reason in decision.get("policy_reasons") or []:
        lines.append(f"Policy note: {reason}")
    zones = vision_result.get("zone_activations")
    if isinstance(zones, dict) and zones:
        zone_str = ", ".join(f"{k}={v:.0%}" for k, v in zones.items())
        lines.append(f"Zone activations (unvalidated occlusion sensitivity, not anatomy): {zone_str}")
        dominant = vision_result.get("dominant_zone")
        if dominant:
            lines.append(f"Dominant occlusion quadrant: {dominant}")
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
- Emphasize that screening is not diagnosis and that symptomatic patients need clinical follow-up even after a negative screen.

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
    has_disclaimer = any(
        "screening" in c.lower() or "diagnosis" in c.lower() or "decision support" in c.lower()
        for c in cautions
    )
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
    raw_sources = _str_list(obj.get("sources"))
    sources = [s for s in raw_sources if s in allowed] if allowed is not None else raw_sources

    cautions = _str_list(obj.get("cautions"))
    has_disclaimer = any("screening" in c.lower() or "diagnosis" in c.lower() for c in cautions)
    if not has_disclaimer:
        cautions.append(
            "This is an AI screening tool, not a definitive diagnosis. Confirmatory testing is required."
        )

    return {
        "risk_level": risk_level,
        "triage": triage,
        "interpretation": str(obj.get("interpretation", "")).strip(),
        "recommendation": str(obj.get("recommendation", "")).strip(),
        "education": _str_list(obj.get("education")),
        "cautions": cautions,
        "sources": sources,
    }
