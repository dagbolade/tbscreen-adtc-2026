# Technical Report — TBScreen

**Team ID:** 1064863
**Domain:** healthcare_medical
**Model:** gemma-4-E2B-it-Q4_K_M (llama.cpp / GGUF Q4_K_M, ~2.6B effective)

## Problem

Community health workers in high-burden TB settings often have chest X-ray access without a radiologist on site. Cloud clinical assistants are blocked by connectivity and cost. TBScreen is an offline decision-support assistant that screens a CXR locally, retrieves WHO-aligned guidance, and returns structured triage language a CHW can act on in English, Yorùbá, Hausa, or Igbo. It is screening support, not diagnosis.

## Design Decisions

**Load-bearing CV + LLM integration.** A MobileNetV3-Small ONNX screener (~4 MB) produces a TB probability and optional 2x2 occlusion-sensitivity weights. The LLM never sees pixels; it consumes the structured vision result plus offline TF-IDF passages. A separate grounded Q&A path serves the metadata test prompts without an image.

**Model selection.** Gemma-4-E2B Q4_K_M is the commercial submission model. Portability tests rejected Qwen3.5-4B because the `qwen35` hybrid SSM architecture does not load in standard `llama-cpp-python` (disqualification risk under llama.cpp-only judging). Qwen 2.5 3B and Llama 3.2 3B were not re-downloaded for this iteration; prior transcript comparisons and license/localization tradeoffs were accepted, and Tiny Aya Earth/Global remain research controls only (CC-BY-NC). On the frozen four-language harness (`data/eval/bakeoff_cases.json`), Gemma scored parse/triage/citation/safety rates of 1.0, mean latency 13.8 s/case, peak RSS 3.13 GB (participant M1 Pro).

**Quantization and templating.** Q4_K_M is used for size and RAM headroom under the 7 GB ceiling. Inference uses each GGUF's chat-completion path rather than a hard-coded Gemma wrapper. `download_model.sh` pins SHA-256 `9378bc47…8672d`.

**Offline RAG and safety policy.** Passages carry language-independent `topic` ids with provenance metadata. Retrieval filters by language and topic so Yorùbá/Hausa/Igbo screens are not queried with English-only risk strings. Unicode tokenization preserves African Latin diacritics. Triage is WHO-aligned decision support: high CAD score leads to refer; borderline leads to retest/refer with symptoms; a negative screen does **not** clear a symptomatic or high-risk patient (`symptom_followup`); CAD workflow is out of scope for age under 15. Occlusion zones are labeled as unvalidated quadrant sensitivity, not anatomy.

## Constraints

| Constraint | Mitigation |
|---|---|
| 7 GB RAM / no discrete GPU | Q4_K_M GGUF; ONNX CPU; lazy LLM load; TF-IDF index (no embedding server) |
| 100% offline inference | Bundled ONNX + GGUF + `corpus/index/`; no CDN fonts; uploads deleted after analyze |
| llama.cpp + GGUF only | Verified load path; rejected non-portable hybrid architectures |
| Clinical safety | Policy engine + citation sanitization + mandatory screening disclaimer |
| African-language bonus | Localized passages + four-language UI/eval; native clinician sign-off still pending (`data/eval/native_review_pending.json`) |

## Benchmarks

**Official profiler (participant laptop, not the ADTC Standard Laptop).** From `submission.json` via `adtc-profiler 0.1.0` on Apple M1 Pro 16 GB at commit `f9f9de8`: 16.73 tok/s generation (512-prompt / 128-gen llama-bench), 7.16 s first-token latency for a 512-token prompt, peak RSS 2135 MB, no thermal throttling. Self-reported scores per the profiler formulas: Sperf = min(16.73/15.0, 1.0)×100 = **100.0** (capped) and Seff = (7.0−2.135)/7.0×100 = **69.5**. These are participant-machine numbers; the audit re-measures on the 4-core / 8 GB sandbox.

**Full application profile (same M1 Pro participant machine).** Vision 0.56 s; LLM cold load 6.65 s; warm screen+RAG+LLM 23.7 s; Q&A 16.7 s; peak RSS ~3.14 GB. Screen and Q&A YAML both parsed.

**Vision smoke (not validation).** `samples/` Shenzhen 20-image set at threshold 0.65: sensitivity 100%, specificity 60%. No AUC is claimed in this report; no held-out patient-level validation artifact is currently committed.

**Harness.** `scripts/eval.py` and `scripts/bakeoff.py` exercise the frozen cases (schema, triage policy, citation faithfulness, safety text) across en/yo/ha/ig. Automated unit tests cover Unicode retrieval, topic RAG, ONNX smoke, and metadata prompt paths.

**Limitations.** (1) Target-laptop profiler output is still outstanding — participant numbers above are from an Apple M1 Pro, not the Ubuntu 22.04 / 8 GB ADTC Standard Laptop. (2) Localized corpus passages are paraphrases pending native clinician approval. (3) African-language quality for the bonus should be treated as harness-supported, not clinician-verified, until `native_review_pending.json` is completed. (4) `team_id` 1064863 now flows correctly into `submission.json`; confirm it against the ADTF/DevPost registration at submit time.
