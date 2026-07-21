#!/usr/bin/env python3
"""llama.cpp (GGUF) runtime — CPU-only inference sized for the ADTC 7 GB target."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Iterable

from llama_cpp import Llama

from . import core

logger = logging.getLogger("tbscreen.llm")

DEFAULT_MODEL = os.environ.get(
    "TBSCREEN_GGUF",
    os.path.join(os.path.dirname(__file__), "..", "..", "model", "gemma-4-E2B-it-Q4_K_M.gguf"),
)

# llama.cpp contexts are not thread-safe; Flask must serialize all generate calls.
_INFER_LOCK = threading.Lock()


def load(model_path: str = DEFAULT_MODEL, n_ctx: int = 2048, n_threads: int | None = None) -> Llama:
    """Load the GGUF model. Single-thread CPU mirrors ADTC and avoids ggml race aborts."""
    threads = 1 if n_threads is None else max(1, int(n_threads))
    logger.info("Loading GGUF path=%s n_ctx=%s n_threads=%s", model_path, n_ctx, threads)
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=0,
        n_threads=threads,
        n_batch=256,
        verbose=False,
    )


def _wrap_chat(user: str) -> str:
    """Gemma turn markers — more stable than jinja chat_completion on this GGUF."""
    return f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"


def _generate(llm: Llama, user: str, max_tokens: int, temperature: float) -> str:
    """Serialize generation; prefer explicit Gemma template to avoid ggml set_rows aborts."""
    prompt = _wrap_chat(user)
    with _INFER_LOCK:
        logger.info("LLM generate start max_tokens=%s prompt_chars=%s", max_tokens, len(prompt))
        try:
            out = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<end_of_turn>", "<|im_end|>"],
            )
            text = out["choices"][0]["text"]
            logger.info("LLM generate done out_chars=%s", len(text or ""))
            return text
        except Exception:
            logger.exception("LLM completion failed; trying chat_completion fallback")
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return out["choices"][0]["message"]["content"]


def interpret(
    llm: Llama,
    vision_result: dict[str, Any],
    context: str,
    lang: str = "English",
    valid_sources: Iterable[str] | None = None,
    max_tokens: int = 512,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Generate a clinical interpretation of a TB screening result, grounded in retrieved passages."""
    decision = decision or core.decide_triage(vision_result.get("tb_probability", 0.0))
    screening_summary = core.build_screening_summary(vision_result, decision)

    prompt = core.interpret_user_content(screening_summary, context, lang)
    raw = _generate(llm, prompt, max_tokens, temperature=0.2)
    parsed = core.parse_yaml(raw)
    return core.sanitize_interpretation(
        parsed,
        decision["risk_level"],
        decision["triage"],
        valid_sources=valid_sources,
    )


def answer_question(
    llm: Llama,
    question: str,
    context: str,
    lang: str = "English",
    valid_sources: Iterable[str] | None = None,
    max_tokens: int = 512,
) -> dict[str, Any] | None:
    """Generate a RAG-grounded clinical Q&A response (no vision input)."""
    prompt = core.qa_user_content(question, context, lang)
    raw = _generate(llm, prompt, max_tokens, temperature=0.2)
    parsed = core.parse_yaml(raw)
    return core.sanitize_qa(parsed, valid_sources=valid_sources)
