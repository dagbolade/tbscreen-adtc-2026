#!/usr/bin/env python3
"""Automated semantic field-level evaluation harness for TBScreen clinical assistant."""

import os
import sys
import json
import time

# Add repository root and src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tbscreen import pipeline, load, Retriever

GGUF_PATH = "/Users/m1pro/.ollama/models/blobs/sha256-9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"
CORPUS_PATH = "corpus/sources/who_tb_guidelines.jsonl"


def run_evaluation():
    print("=" * 60)
    print("TBSCREEN SEMANTIC EVALUATION HARNESS")
    print("=" * 60)

    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus {CORPUS_PATH} not found.")
        sys.exit(1)

    print("Loading LLM context...")
    t0 = time.time()
    llm_model = load(model_path=GGUF_PATH)
    print(f"LLM loaded in {time.time() - t0:.2f}s")

    print("Loading RAG Database...")
    retriever = Retriever.from_jsonl(CORPUS_PATH)
    valid_source_ids = {p["id"] for p in retriever.passages}
    print(f"Loaded {len(valid_source_ids)} passages from RAG.")

    # 3 Evaluation scenarios
    scenarios = [
        {
            "name": "High Risk Case",
            "vision_result": {"tb_probability": 0.82, "screening_result": "POSITIVE"},
            "expected_triage": "refer",
            "expected_risk": "high"
        },
        {
            "name": "Moderate Risk Case",
            "vision_result": {"tb_probability": 0.52, "screening_result": "BORDERLINE"},
            "expected_triage": "retest",
            "expected_risk": "moderate"
        },
        {
            "name": "Low Risk Case",
            "vision_result": {"tb_probability": 0.22, "screening_result": "NEGATIVE"},
            "expected_triage": "clear",
            "expected_risk": "low"
        }
    ]

    languages = ["English", "Yoruba", "Hausa", "Igbo"]

    results = []
    all_passed = True

    for scenario in scenarios:
        print(f"\nEvaluating: {scenario['name']}")
        print("-" * 50)
        
        for lang in languages:
            print(f"  Language: {lang}...", end="", flush=True)
            
            t_start = time.time()
            out = pipeline.screen_and_interpret(
                model=llm_model,
                retriever=retriever,
                vision_result=scenario["vision_result"],
                lang=lang,
                k=3
            )
            elapsed = time.time() - t_start
            
            # Validation checks
            passed = True
            errors = []
            
            interpret = out.get("interpretation")
            if not interpret:
                passed = False
                errors.append("YAML parsing failed (None output)")
            else:
                # 1. Field validation
                for field in ["interpretation", "recommendation", "education", "cautions", "sources"]:
                    if field not in interpret:
                        passed = False
                        errors.append(f"Missing field: {field}")
                
                # 2. Triage consistency
                if out.get("triage") != scenario["expected_triage"]:
                    passed = False
                    errors.append(f"Triage mismatch: expected={scenario['expected_triage']}, got={out.get('triage')}")
                
                # 3. Risk consistency
                if out.get("risk_level") != scenario["expected_risk"]:
                    passed = False
                    errors.append(f"Risk mismatch: expected={scenario['expected_risk']}, got={out.get('risk_level')}")
                
                # 4. Anti-hallucination check (sources constraint)
                sources = interpret.get("sources", [])
                for src in sources:
                    if src not in valid_source_ids:
                        passed = False
                        errors.append(f"Hallucinated citation: {src}")
                
                # 5. Disclaimer presence check
                cautions = interpret.get("cautions", [])
                has_disclaimer = any("screening" in c.lower() or "diagnosis" in c.lower() for c in cautions)
                if not has_disclaimer:
                    passed = False
                    errors.append("Safety disclaimer missing from cautions list")
                    
            if passed:
                print(" PASSED")
            else:
                print(" FAILED")
                all_passed = False
                for err in errors:
                    print(f"    - {err}")
                    
            results.append({
                "scenario": scenario["name"],
                "language": lang,
                "passed": passed,
                "errors": errors,
                "elapsed_sec": elapsed
            })

    # Print Summary Report
    print("\n" + "=" * 60)
    print("SUMMARY EVALUATION REPORT")
    print("=" * 60)
    
    total = len(results)
    passed_cnt = sum(1 for r in results if r["passed"])
    print(f"Total Test Runs: {total}")
    print(f"Passed: {passed_cnt} / {total} ({passed_cnt/total:.1%})")
    
    avg_latency = sum(r["elapsed_sec"] for r in results) / total
    print(f"Average Pipeline Latency: {avg_latency:.2f} seconds")
    
    print("\nVerdict:", "SUCCESS" if all_passed else "FAILURE")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
