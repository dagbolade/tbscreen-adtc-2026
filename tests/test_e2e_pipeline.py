#!/usr/bin/env python3
"""E2E Pipeline Test: runs vision screening, retrieval, and clinical interpretation."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tbscreen import TBScreenAssistant
from tbscreen.core import dump_yaml


def main():
    image_path = os.path.join(ROOT, "tests", "test_xray.jpg")
    if not os.path.exists(image_path):
        print(f"Error: test image {image_path} not found.")
        sys.exit(1)

    gguf_path = os.environ.get(
        "TBSCREEN_GGUF",
        os.path.join(ROOT, "model", "gemma-4-E2B-it-Q4_K_M.gguf"),
    )
    if not os.path.exists(gguf_path):
        print(f"Error: GGUF not found at {gguf_path}")
        print("Run: bash download_model.sh  (or set TBSCREEN_GGUF)")
        sys.exit(1)

    print("Initializing TBScreenAssistant...")
    assistant = TBScreenAssistant(
        llm_model_path=gguf_path,
        corpus_path=os.path.join(ROOT, "corpus", "sources", "who_tb_guidelines.jsonl"),
    )

    print("\n--------------------------------------------------")
    print("Running E2E process (English)...")
    print("--------------------------------------------------")
    result_en = assistant.process_image(image_path, lang="English")
    print("Vision Screening Result:")
    print(json.dumps(result_en["vision_result"], indent=2))
    print(f"Risk Level: {result_en['risk_level']} | Triage Action: {result_en['triage']}")
    print("Retrieved Source IDs:", result_en["retrieved_sources"])
    print("\nLLM Interpretation (Grounded):")
    if result_en["interpretation"]:
        print(dump_yaml(result_en["interpretation"]))
    else:
        print("Interpretation failed.")

    print("\n--------------------------------------------------")
    print("Running language switch via cached vision (Yoruba)...")
    print("--------------------------------------------------")
    result_yo = assistant.reinterpret(lang="Yoruba")
    print("LLM Interpretation (Yoruba):")
    if result_yo["interpretation"]:
        print(dump_yaml(result_yo["interpretation"]))
    else:
        print("Interpretation failed.")

    print("\n--------------------------------------------------")
    print("Running clinical Q&A path...")
    print("--------------------------------------------------")
    qa = assistant.ask(
        "What are the key differences between a positive TB screening result "
        "and a confirmed TB diagnosis?"
    )
    if qa["answer"]:
        print(dump_yaml(qa["answer"]))
    else:
        print("Q&A failed.")


if __name__ == "__main__":
    main()
