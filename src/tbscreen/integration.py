"""Integration bridge: binds the ONNX vision model with the RAG+LLM pipeline."""

from __future__ import annotations

import os
from typing import Any

from llama_cpp import Llama

from vision.inference import TBScreenModel
from . import pipeline
from .rag import Retriever


class TBScreenAssistant:
    """Coordinated assistant pairing the ONNX vision model with the RAG + LLM pipeline."""

    def __init__(
        self,
        llm_model_path: str | None = None,
        vision_model_path: str | None = None,
        corpus_path: str | None = None,
    ):
        here = os.path.dirname(__file__)
        if corpus_path is None:
            corpus_path = os.path.join(here, "..", "..", "corpus", "sources", "who_tb_guidelines.jsonl")

        self.vision_model = TBScreenModel(model_path=vision_model_path)

        if os.path.exists(corpus_path):
            self.retriever = Retriever.from_jsonl(corpus_path)
        else:
            fixture_path = os.path.join(here, "..", "..", "tests", "fixture_corpus.jsonl")
            self.retriever = Retriever.from_jsonl(fixture_path)

        self.llm_model_path = llm_model_path
        self._llm: Llama | None = None
        self._last_vision_result: dict[str, Any] | None = None
        self._last_image_path: str | None = None
        self._last_patient_context: dict[str, Any] | None = None

    @property
    def llm(self) -> Llama:
        """Lazy load LLM context to preserve memory until needed."""
        if self._llm is None:
            from . import llm

            if self.llm_model_path:
                self._llm = llm.load(model_path=self.llm_model_path)
            else:
                self._llm = llm.load()
        return self._llm

    def clear_session(self) -> None:
        """Drop cached vision/patient state so sessions cannot leak results."""
        self._last_vision_result = None
        self._last_image_path = None
        self._last_patient_context = None

    def process_image(
        self,
        image_path: str,
        lang: str = "English",
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Screen a chest X-ray and interpret with WHO RAG context."""
        if self._last_image_path == image_path and self._last_vision_result is not None:
            vision_result = self._last_vision_result
        else:
            # UI path skips occlusion zones (faster; avoids oversized prompts).
            vision_result = self.vision_model.predict(image_path, with_zones=False)
            self._last_image_path = image_path
            self._last_vision_result = vision_result

        if patient_context is not None:
            self._last_patient_context = dict(patient_context)

        return pipeline.screen_and_interpret(
            model=self.llm,
            retriever=self.retriever,
            vision_result=vision_result,
            lang=lang,
            k=3,
            patient_context=self._last_patient_context,
        )

    def reinterpret(self, lang: str = "English") -> dict[str, Any]:
        """Re-run LLM+RAG on the cached vision result (language toggle)."""
        if self._last_vision_result is None:
            raise ValueError("No cached vision result — analyze an image first")
        return pipeline.screen_and_interpret(
            model=self.llm,
            retriever=self.retriever,
            vision_result=self._last_vision_result,
            lang=lang,
            k=3,
            patient_context=self._last_patient_context,
        )

    def ask(self, question: str, lang: str = "English") -> dict[str, Any]:
        """Grounded clinical Q&A; personalized when a screening result is cached."""
        return pipeline.answer_clinical_question(
            model=self.llm,
            retriever=self.retriever,
            question=question,
            lang=lang,
            k=4,
            vision_result=self._last_vision_result,
            patient_context=self._last_patient_context,
        )
