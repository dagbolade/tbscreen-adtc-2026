"""Build offline TF-IDF RAG index from the WHO guidelines corpus."""

from __future__ import annotations

import os
import sys

import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tbscreen.rag import Retriever  # noqa: E402

CORPUS = os.path.join(ROOT, "corpus", "sources", "who_tb_guidelines.jsonl")
INDEX_DIR = os.path.join(ROOT, "corpus", "index")


def main() -> None:
    passages = [json.loads(line) for line in open(CORPUS, encoding="utf-8") if line.strip()]
    retriever = Retriever.from_passages(passages)
    retriever.save_index(INDEX_DIR)
    print(f"Wrote TF-IDF index for {len(retriever.passages)} passages → {INDEX_DIR}")


if __name__ == "__main__":
    main()
