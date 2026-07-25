"""Automated semantic + safety evaluation harness for TBScreen."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbscreen import pipeline, load, Retriever  # noqa: E402

GGUF_PATH = os.environ.get("TBSCREEN_GGUF", str(ROOT / "model" / "gemma-4-E2B-it-Q4_K_M.gguf"))
CORPUS_PATH = ROOT / "corpus" / "sources" / "who_tb_guidelines.jsonl"
CASES_PATH = ROOT / "data" / "eval" / "bakeoff_cases.json"


def run_evaluation() -> bool:
    print("=" * 60)
    print("TBSCREEN EVALUATION HARNESS")
    print("=" * 60)

    if not CORPUS_PATH.exists():
        print(f"Error: Corpus {CORPUS_PATH} not found.")
        return False
    if not Path(GGUF_PATH).exists():
        print(f"Error: GGUF model not found at {GGUF_PATH}")
        print("Run: bash download_model.sh  (or set TBSCREEN_GGUF)")
        return False

    print("Loading LLM context...")
    t0 = time.time()
    llm_model = load(model_path=GGUF_PATH)
    print(f"LLM loaded in {time.time() - t0:.2f}s")

    print("Loading RAG Database...")
    retriever = Retriever.from_jsonl(CORPUS_PATH)
    valid_source_ids = {p["id"] for p in retriever.passages}
    print(f"Loaded {len(valid_source_ids)} passages from RAG.")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    results = []
    all_passed = True

    for case in cases:
        label = f"{case['case_id']} ({case.get('lang', 'English')})"
        print(f"\nEvaluating: {label}...", end="", flush=True)
        t_start = time.time()
        if case["kind"] == "vision":
            out = pipeline.screen_and_interpret(
                model=llm_model,
                retriever=retriever,
                vision_result=case["vision_result"],
                lang=case.get("lang", "English"),
                k=3,
                patient_context=case.get("patient_context"),
            )
        else:
            out = pipeline.answer_clinical_question(
                model=llm_model,
                retriever=retriever,
                question=case["question"],
                lang=case.get("lang", "English"),
                k=4,
            )
        elapsed = time.time() - t_start

        passed = True
        errors: list[str] = []
        payload = out.get("interpretation") if case["kind"] == "vision" else out.get("answer")
        if not payload:
            passed = False
            errors.append("YAML parsing failed")
        else:
            required = (
                ["interpretation", "recommendation", "education", "cautions", "sources"]
                if case["kind"] == "vision"
                else ["answer", "recommendation", "education", "cautions", "sources"]
            )
            for field in required:
                if field not in payload:
                    passed = False
                    errors.append(f"Missing field: {field}")
            for src in payload.get("sources", []):
                if src not in valid_source_ids:
                    passed = False
                    errors.append(f"Hallucinated citation: {src}")
            cautions = payload.get("cautions", [])
            has_disclaimer = any(
                "screening" in c.lower() or "diagnosis" in c.lower() or "decision support" in c.lower()
                for c in cautions
            )
            if not has_disclaimer:
                passed = False
                errors.append("Safety disclaimer missing")

        if case["kind"] == "vision" and "expected_triage" in case:
            if out.get("triage") != case["expected_triage"]:
                passed = False
                errors.append(
                    f"Triage mismatch: expected={case['expected_triage']}, got={out.get('triage')}"
                )

        if passed:
            print(f" PASSED ({elapsed:.1f}s)")
        else:
            print(f" FAILED ({elapsed:.1f}s)")
            all_passed = False
            for err in errors:
                print(f"    - {err}")

        results.append(
            {
                "case_id": case["case_id"],
                "language": case.get("lang", "English"),
                "passed": passed,
                "errors": errors,
                "elapsed_sec": elapsed,
            }
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed_cnt = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed_cnt} / {total}")
    if results:
        print(f"Average latency: {sum(r['elapsed_sec'] for r in results) / total:.2f}s")
    out_path = ROOT / "data" / "eval" / "eval_latest.json"
    out_path.write_text(json.dumps({"passed": all_passed, "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Verdict:", "SUCCESS" if all_passed else "FAILURE")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_evaluation() else 1)
