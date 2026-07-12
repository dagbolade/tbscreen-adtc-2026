#!/usr/bin/env python3
"""llama.cpp (GGUF) runtime — CPU-only inference sized for the ADTC 7 GB target."""

from __future__ import annotations

import os
from typing import Any, Iterable

from llama_cpp import Llama

from . import core

# Submission uses model/ (see download_model.sh).
DEFAULT_MODEL = os.environ.get(
    "TBSCREEN_GGUF",
    os.path.join(os.path.dirname(__file__), "..", "..", "model", "gemma-4-E2B-it-Q4_K_M.gguf"),
)


def load(model_path: str = DEFAULT_MODEL, n_ctx: int = 4096, n_threads: int | None = None) -> Llama:
    """Load the GGUF model. n_gpu_layers=0 mirrors the ADTC target (no discrete GPU)."""
    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=0, n_threads=n_threads, verbose=False)


def _gemma_prompt(user: str) -> str:
    """Gemma-4 chat template."""
    return f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"


def _generate(llm: Llama, user: str, max_tokens: int, temperature: float) -> str:
    out = llm(_gemma_prompt(user), max_tokens=max_tokens, temperature=temperature, stop=["<end_of_turn>"])
    return out["choices"][0]["text"]


def interpret(
    llm: Llama,
    vision_result: dict[str, Any],
    context: str,
    lang: str = "English",
    valid_sources: Iterable[str] | None = None,
    max_tokens: int = 600,
) -> dict[str, Any] | None:
    """Generate a clinical interpretation of a TB screening result, grounded in retrieved passages."""
    screening_summary = core.build_screening_summary(vision_result)
    risk_level, triage = core.classify_risk(vision_result.get("tb_probability", 0.0))

    prompt = core.interpret_user_content(screening_summary, context, lang)
    raw = _generate(llm, prompt, max_tokens, temperature=0.2)
    parsed = core.parse_yaml(raw)
    return core.sanitize_interpretation(parsed, risk_level, triage, valid_sources=valid_sources)
