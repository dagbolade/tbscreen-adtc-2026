#!/usr/bin/env python3
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
        # Default paths
        here = os.path.dirname(__file__)
        if corpus_path is None:
            corpus_path = os.path.join(here, "..", "..", "corpus", "sources", "who_tb_guidelines.jsonl")
        
        # Load the ONNX vision model
        self.vision_model = TBScreenModel(model_path=vision_model_path)
        
        # Load the RAG database/retriever
        if os.path.exists(corpus_path):
            self.retriever = Retriever.from_jsonl(corpus_path)
        else:
            # Fallback to test fixtures if main corpus doesn't exist
            fixture_path = os.path.join(here, "..", "..", "tests", "fixture_corpus.jsonl")
            self.retriever = Retriever.from_jsonl(fixture_path)
            
        # Keep track of model path to lazy-load Llama context
        self.llm_model_path = llm_model_path
        self._llm: Llama | None = None

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

    def process_image(self, image_path: str, lang: str = "English") -> dict[str, Any]:
        """Screen a chest X-ray and interpret the results with the LLM and WHO RAG context."""
        # 1. Run vision screening
        vision_result = self.vision_model.predict(image_path)
        
        # 2. Run LLM interpretation grounded in WHO TB guidelines
        result = pipeline.screen_and_interpret(
            model=self.llm,
            retriever=self.retriever,
            vision_result=vision_result,
            lang=lang,
            k=3  # Retrieve top 3 relevant passages
        )
        return result
