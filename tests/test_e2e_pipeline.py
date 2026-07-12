#!/usr/bin/env python3
"""E2E Pipeline Test: runs vision screening, retrieval, and clinical interpretation."""

import sys
import os

# Add repository root and src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tbscreen import TBScreenAssistant
from tbscreen.core import dump_yaml

def main():
    image_path = "tests/test_xray.jpg"
    if not os.path.exists(image_path):
        print(f"Error: test image {image_path} not found.")
        sys.exit(1)
        
    print("Initializing TBScreenAssistant...")
    # Points to direct GGUF model path in Ollama blobs for development
    gguf_path = "/Users/m1pro/.ollama/models/blobs/sha256-9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"
    
    assistant = TBScreenAssistant(
        llm_model_path=gguf_path,
        corpus_path="corpus/sources/who_tb_guidelines.jsonl"
    )
    
    print("\n--------------------------------------------------")
    print("Running E2E process (English)...")
    print("--------------------------------------------------")
    result_en = assistant.process_image(image_path, lang="English")
    print("Vision Screening Result:")
    print(json_format(result_en["vision_result"]))
    print(f"Risk Level: {result_en['risk_level']} | Triage Action: {result_en['triage']}")
    print("Retrieved Source IDs:", result_en["retrieved_sources"])
    print("\nLLM Interpretation (Grounded):")
    if result_en["interpretation"]:
        print(dump_yaml(result_en["interpretation"]))
    else:
        print("Interpretation failed.")

    print("\n--------------------------------------------------")
    print("Running E2E process (Yoruba)...")
    print("--------------------------------------------------")
    result_yo = assistant.process_image(image_path, lang="Yoruba")
    print("LLM Interpretation (Yoruba):")
    if result_yo["interpretation"]:
        print(dump_yaml(result_yo["interpretation"]))
    else:
        print("Interpretation failed.")

def json_format(d):
    import json
    return json.dumps(d, indent=2)

if __name__ == "__main__":
    main()
