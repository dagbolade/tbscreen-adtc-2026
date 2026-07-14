"""TBScreen — offline, RAG-grounded TB clinical assistant on llama.cpp."""

from .integration import TBScreenAssistant
from .pipeline import screen_and_interpret, answer_clinical_question
from .rag import Retriever
from .llm import load, interpret, answer_question

__all__ = [
    "TBScreenAssistant",
    "screen_and_interpret",
    "answer_clinical_question",
    "Retriever",
    "load",
    "interpret",
    "answer_question",
]
