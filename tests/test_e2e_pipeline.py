"""E2E pipeline test using a sample CXR."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbscreen import TBScreenAssistant  # noqa: E402
from tbscreen.core import dump_yaml  # noqa: E402


def main() -> int:
    # Prefer tracked Shenzhen samples; fall back to tests/test_xray.jpg if present.
    candidates = [
        ROOT / "samples" / "normal_01.png",
        ROOT / "samples" / "tb_01.png",
        ROOT / "tests" / "test_xray.jpg",
    ]
    image_path = next((p for p in candidates if p.exists()), None)
    if image_path is None:
        print("Error: no tracked sample image found under samples/")
        return 1

    gguf_path = Path(os.environ.get("TBSCREEN_GGUF", str(ROOT / "model" / "gemma-4-E2B-it-Q4_K_M.gguf")))
    if not gguf_path.exists():
        print(f"Error: GGUF not found at {gguf_path}")
        print("Run: bash download_model.sh  (or set TBSCREEN_GGUF)")
        return 1

    print(f"Using image: {image_path}")
    assistant = TBScreenAssistant(
        llm_model_path=str(gguf_path),
        corpus_path=str(ROOT / "corpus" / "sources" / "who_tb_guidelines.jsonl"),
    )

    print("\n--- English screen ---")
    result_en = assistant.process_image(
        str(image_path),
        lang="English",
        patient_context={"age_years": 34, "cough_weeks": 1, "has_tb_symptoms": False},
    )
    print(json.dumps(result_en["vision_result"], indent=2))
    print(f"Risk={result_en['risk_level']} triage={result_en['triage']}")
    print("Sources:", result_en["retrieved_sources"])
    if result_en["interpretation"]:
        print(dump_yaml(result_en["interpretation"]))

    print("\n--- Yoruba reinterpret ---")
    result_yo = assistant.reinterpret(lang="Yoruba")
    if result_yo["interpretation"]:
        print(dump_yaml(result_yo["interpretation"]))

    print("\n--- Clinical Q&A ---")
    qa = assistant.ask(
        "What are the key differences between a positive TB screening result "
        "and a confirmed TB diagnosis?"
    )
    if qa["answer"]:
        print(dump_yaml(qa["answer"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
