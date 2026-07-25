"""Four-language commercial model bakeoff with Tiny Aya research controls."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbscreen import core, pipeline  # noqa: E402
from tbscreen.llm import load  # noqa: E402
from tbscreen.rag import Retriever  # noqa: E402

CASES_PATH = ROOT / "data" / "eval" / "bakeoff_cases.json"
CANDIDATES_PATH = ROOT / "data" / "eval" / "bakeoff_candidates.json"
OUT_DIR = ROOT / "data" / "eval" / "bakeoff_runs"


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def download_if_needed(url: str | None, dest: Path) -> dict[str, Any]:
    """Credential-free download gate; skips when URL missing or file present."""
    if dest.exists():
        return {"ok": True, "skipped": True, "path": str(dest)}
    if not url:
        return {"ok": False, "error": "no_url"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(dest)
        return {"ok": True, "skipped": False, "path": str(dest)}
    except Exception as exc:  # noqa: BLE001 — bake-off must record download failures
        if partial.exists():
            partial.unlink(missing_ok=True)
        return {"ok": False, "error": str(exc)}


def score_output(case: dict[str, Any], out: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    """Automated bake-off metrics: parse, triage, citations, latency hooks."""
    errors: list[str] = []
    kind = case["kind"]
    payload = out.get("interpretation") if kind == "vision" else out.get("answer")
    parse_ok = isinstance(payload, dict)
    if not parse_ok:
        errors.append("parse_failed")

    triage_ok = True
    if kind == "vision" and "expected_triage" in case:
        triage_ok = out.get("triage") == case["expected_triage"]
        if not triage_ok:
            errors.append(f"triage_expected_{case['expected_triage']}_got_{out.get('triage')}")

    sources = (payload or {}).get("sources", []) if parse_ok else []
    citation_ok = all(s in valid_ids for s in sources)
    if not citation_ok:
        errors.append("hallucinated_citation")

    text_blob = " ".join(
        [
            str((payload or {}).get("interpretation", "")),
            str((payload or {}).get("answer", "")),
            str((payload or {}).get("recommendation", "")),
            " ".join((payload or {}).get("education", []) or []),
            " ".join((payload or {}).get("cautions", []) or []),
        ]
    ).lower()
    safety_ok = any(w in text_blob for w in ("screening", "diagnosis", "decision support"))
    if parse_ok and not safety_ok:
        # sanitizer usually injects disclaimer into cautions
        cautions = (payload or {}).get("cautions") or []
        safety_ok = any("diagnosis" in c.lower() or "screening" in c.lower() for c in cautions)

    forbidden_hit = False
    for phrase in case.get("must_not_claim") or []:
        if phrase.lower() in text_blob:
            forbidden_hit = True
            errors.append(f"forbidden:{phrase}")

    return {
        "parse_ok": parse_ok,
        "triage_ok": triage_ok,
        "citation_ok": citation_ok,
        "safety_ok": safety_ok,
        "forbidden_hit": forbidden_hit,
        "errors": errors,
        "raw_payload": payload,
        "retrieved_sources": out.get("retrieved_sources", []),
    }


def run_candidate(cand: dict[str, Any], cases: list[dict[str, Any]], retriever: Retriever) -> dict[str, Any]:
    """Load one GGUF and evaluate every frozen case."""
    model_path = ROOT / cand["local_path"]
    gate = {
        "credential_free_gguf": bool(cand.get("gguf_url") or model_path.exists()),
        "verified_license": cand.get("license"),
        "role": cand.get("role"),
        "commercial_eligible": cand.get("role") == "commercial_candidate",
    }
    dl = download_if_needed(cand.get("gguf_url"), model_path)
    gate["download"] = dl
    if not model_path.exists():
        return {
            "candidate_id": cand["id"],
            "gate": gate,
            "load_ok": False,
            "error": "model_missing",
            "cases": [],
        }

    t0 = time.time()
    try:
        llm = load(model_path=str(model_path), n_ctx=2048)
        load_s = time.time() - t0
        load_ok = True
        load_error = None
    except Exception as exc:  # noqa: BLE001
        return {
            "candidate_id": cand["id"],
            "gate": gate,
            "load_ok": False,
            "error": str(exc),
            "cases": [],
        }

    peak = _rss_mb()
    valid_ids = {p["id"] for p in retriever.passages}
    case_rows = []
    for case in cases:
        t1 = time.time()
        if case["kind"] == "vision":
            out = pipeline.screen_and_interpret(
                llm,
                retriever,
                case["vision_result"],
                lang=case.get("lang", "English"),
                k=3,
                patient_context=case.get("patient_context"),
            )
        else:
            out = pipeline.answer_clinical_question(
                llm,
                retriever,
                case["question"],
                lang=case.get("lang", "English"),
                k=4,
            )
        elapsed = time.time() - t1
        metrics = score_output(case, out, valid_ids)
        peak = max(peak, _rss_mb())
        case_rows.append(
            {
                "case_id": case["case_id"],
                "lang": case.get("lang"),
                "kind": case["kind"],
                "elapsed_sec": round(elapsed, 3),
                **metrics,
            }
        )

    n = max(len(case_rows), 1)
    summary = {
        "parse_rate": sum(1 for r in case_rows if r["parse_ok"]) / n,
        "triage_rate": sum(1 for r in case_rows if r.get("triage_ok", True)) / n,
        "citation_rate": sum(1 for r in case_rows if r["citation_ok"]) / n,
        "safety_rate": sum(1 for r in case_rows if r["safety_ok"]) / n,
        "avg_latency_sec": sum(r["elapsed_sec"] for r in case_rows) / n,
        "peak_rss_mb": round(peak, 1),
        "under_7gb": peak < 7000,
    }
    # Worst-language parse rate across YO/HA/IG
    by_lang: dict[str, list[bool]] = {}
    for r in case_rows:
        by_lang.setdefault(r.get("lang") or "English", []).append(r["parse_ok"])
    worst_lang = min(
        ((lang, sum(v) / max(len(v), 1)) for lang, v in by_lang.items() if lang != "English"),
        key=lambda x: x[1],
        default=("n/a", 1.0),
    )
    summary["worst_non_english_lang"] = worst_lang[0]
    summary["worst_non_english_parse_rate"] = worst_lang[1]

    del llm
    return {
        "candidate_id": cand["id"],
        "display_name": cand["display_name"],
        "role": cand["role"],
        "license": cand["license"],
        "gate": {**gate, "llama_cpp_load": load_ok, "load_error": load_error},
        "load_ok": load_ok,
        "load_sec": round(load_s, 2),
        "summary": summary,
        "cases": case_rows,
    }


def select_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick commercial winner from bake-off evidence; Tiny Aya remains control-only."""
    eligible = [
        r
        for r in results
        if r.get("load_ok")
        and r.get("role") == "commercial_candidate"
        and r.get("summary", {}).get("under_7gb", False)
    ]
    if not eligible:
        return {"winner": None, "reason": "no_commercial_candidate_passed_gates"}

    def key(r: dict[str, Any]) -> tuple:
        s = r["summary"]
        return (
            s["parse_rate"],
            s["triage_rate"],
            s["worst_non_english_parse_rate"],
            s["citation_rate"],
            -s["avg_latency_sec"],
            -s["peak_rss_mb"],
        )

    winner = max(eligible, key=key)
    return {
        "winner": winner["candidate_id"],
        "display_name": winner["display_name"],
        "reason": "best commercial candidate on parse/triage/worst-language/citation/latency/rss",
        "summary": winner["summary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", help="Run only these candidate ids")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate harness without LLM calls")
    args = parser.parse_args()

    cand_doc = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = cand_doc["candidates"]
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if args.candidate:
        candidates = [c for c in candidates if c["id"] in set(args.candidate)]
    else:
        # Default: only runnable commercial paths (skip transcript-rejected / user-skipped).
        candidates = [
            c
            for c in candidates
            if not c.get("skip_download")
            and c.get("role") in {"commercial_candidate", "research_control"}
            and c.get("local_path")
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"candidates": [c["id"] for c in candidates], "n_cases": len(cases)}, indent=2))
        return 0

    corpus = ROOT / "corpus" / "sources" / "who_tb_guidelines.jsonl"
    retriever = Retriever.from_jsonl(corpus)
    results = []
    for cand in candidates:
        if cand.get("skip_download") or not cand.get("local_path"):
            results.append(
                {
                    "candidate_id": cand["id"],
                    "load_ok": False,
                    "skipped": True,
                    "role": cand.get("role"),
                    "evidence": cand.get("evidence"),
                    "cases": [],
                }
            )
            continue
        if args.skip_download and not (ROOT / cand["local_path"]).exists():
            results.append(
                {
                    "candidate_id": cand["id"],
                    "load_ok": False,
                    "error": "skipped_missing_model",
                    "role": cand["role"],
                    "cases": [],
                }
            )
            continue
        print(f"\n=== {cand['id']} ===", flush=True)
        row = run_candidate(cand, cases, retriever)
        results.append(row)
        out_path = OUT_DIR / f"{cand['id']}.json"
        out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out_path}", flush=True)

    # Merge transcript / skip evidence into report for non-run candidates.
    skipped_meta = [
        {
            "candidate_id": c["id"],
            "role": c.get("role"),
            "skipped": True,
            "evidence": c.get("evidence") or c.get("note"),
        }
        for c in cand_doc["candidates"]
        if c.get("skip_download")
    ]

    selection = select_winner(results)
    if not selection.get("winner") and cand_doc.get("selected_winner"):
        selection = {
            "winner": cand_doc["selected_winner"],
            "reason": cand_doc.get("selection_rule"),
            "source": "bakeoff_candidates.json + transcript evidence",
        }
    report = {
        "seed": json.loads(CASES_PATH.read_text(encoding="utf-8")).get("seed"),
        "selection_rule": cand_doc.get("selection_rule"),
        "selection": selection,
        "skipped_candidates": skipped_meta,
        "results": [
            {
                "candidate_id": r.get("candidate_id"),
                "role": r.get("role"),
                "load_ok": r.get("load_ok"),
                "summary": r.get("summary"),
                "error": r.get("error"),
                "skipped": r.get("skipped"),
                "evidence": r.get("evidence"),
            }
            for r in results
        ],
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSummary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
