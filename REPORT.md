# TBScreen: Technical Report

Team 1064863 · healthcare_medical · gemma-4-E2B-it-Q4_K_M (llama.cpp, GGUF Q4_K_M, ~2.6B effective)

## The problem

A community health worker in a high-burden TB setting can often get a chest X-ray taken, but there is no radiologist to read it and no reliable internet to send it anywhere. Cloud-based clinical AI is useless in exactly the places that need it most. We built TBScreen to test a simple premise: a complete screening assistant (one that reads the X-ray, applies triage policy, explains itself in the patient's language, and answers follow-up questions) can run on a basic 8 GB laptop with the network cable unplugged.

To be clear about scope: TBScreen is screening decision support. It is not a diagnostic device, and every output it produces says so.

## Design decisions

The single most important decision we made is that **the LLM never makes a safety decision and never sees a pixel.** A 4.4 MB MobileNetV3-Small (ONNX Runtime, CPU) turns the X-ray into a TB probability. Plain, testable Python turns that probability plus patient context (age, cough duration, symptoms, HIV status, household contact) into a triage action under WHO-aligned rules: high score refers, borderline retests (or refers if symptomatic or high-risk), a negative screen never clears a symptomatic patient, and anyone under 15 is routed out of the CAD workflow entirely. Only after those decisions are locked in does the LLM get involved, and its job is purely language: turning the vision result, the triage decision, and retrieved guideline passages into structured output the health worker can read.

We chose the model by bake-off rather than by reputation. On a frozen four-language harness (`data/eval/bakeoff_cases.json`) we scored candidates on YAML parse rate, triage correctness, citation honesty, and peak memory. Gemma 4 E2B Q4_K_M came out on top with perfect parse/triage/citation/safety rates, 13.8 s mean latency per case, and 3.13 GB peak RSS on our dev machine. The also-rans are worth mentioning because of *why* they lost: Qwen 3.5 4B simply would not load in `llama-cpp-python` (its hybrid SSM architecture is a portability risk under a llama.cpp-only judging rule), and Cohere's Aya models are research-licensed (CC-BY-NC), which rules out a commercial pilot. Qwen 2.5 3B and Llama 3.2 3B were compared on prior transcript evidence and lost on localization and license trade-offs. A model that can't run in the judge's sandbox scores zero no matter how smart it is, so portability weighed as heavily as quality.

The retrieval layer is deliberately boring. 98 WHO-guideline passages (paraphrased, in English, Yorùbá, Hausa, and Igbo) sit in a hand-rolled TF-IDF index: no embedding server, no model download, nothing to break offline. The tokenizer preserves African Latin diacritics so ọ and o are not the same word, and when a query comes in a non-English language we score the English passages too and boost same-topic passages in the target language as a cross-lingual proxy. Retrieval filters by language and topic, so a Yorùbá screen is never answered from English-only risk text.

Because an LLM in a clinical loop will occasionally invent things, we treat its output as untrusted input. Every response must arrive as structured YAML; a sanitizer strips any citation id that does not match a passage actually retrieved for that request, and if the model drops the screening disclaimer, one is injected before the user sees anything. The clinical Q&A path follows the same rules: guideline questions get guideline answers, and if a screening has been run in the session, the answer is grounded in that patient's actual result rather than a generic paragraph.

## Constraints we designed around

The 7 GB RAM ceiling and the llama.cpp-only rule shaped almost everything. Q4_K_M quantization keeps the LLM around 2.9 GB on disk; the vision model is 4.4 MB; the whole stack peaks near 3.1 GB, which leaves real headroom against the disqualification line. Inference is CPU-only with lazy model loading, so the app starts fast and only pays the GGUF load when a generation is actually needed. Offline means offline: system fonts only, no CDN, no telemetry, uploaded images deleted right after analysis. Sessions hold nothing across restarts.

## What we measured

All numbers below are from our participant machine (Apple M1 Pro, 16 GB), not the ADTC Standard Laptop. The audit sandbox will re-measure everything, and we expect throughput to differ there.

Official profiler (`adtc-profiler 0.1.0`, participant mode, at commit `f9f9de8`): 16.73 tok/s generation, 7.16 s first-token latency on a 512-token prompt, 2135 MB peak RSS, no thermal throttling. Feeding those into the profiler's own formulas gives the self-reported Gate-1 numbers: Sperf = min(16.73/15.0, 1.0) × 100 = **100.0** (the cap does the work here) and Seff = (7.0 - 2.135)/7.0 × 100 = **69.5**.

End-to-end application timings on the same machine: vision 0.56 s per image, LLM cold load 6.65 s, a warm screen-plus-interpretation turn 23.7 s, a clinical Q&A turn 16.7 s, full-stack peak RSS about 3.14 GB.

We also validated the whole stack inside an Ubuntu 22.04 container capped at 4 CPUs and 8 GB with swap disabled. The documented install path works there end to end (llama-cpp-python 0.3.30 compiles from source with build-essential and cmake; the pinned numpy and onnxruntime need Python 3.11+, so we used 3.12), all 19 unit tests pass, and a complete screen, reinterpret, and Q&A session finishes with peak RSS of 3.33 GB and zero swapping under the 8 GB cgroup. One caveat we want on the record: container throughput is not evidence of anything. A 4-thread llama-bench inside this virtualized environment reaches about 4 tok/s against 16.73 tok/s natively on the same physical machine, so we report the container run for compatibility and memory safety only, and the target-hardware throughput question stays open until the audit measures it.

On our 20-image Shenzhen smoke set (threshold 0.65) the vision model scores 100% sensitivity and 60% specificity. We label this a smoke test, not validation: it is far too small to claim an AUC, and we don't. Beyond the smoke set, an automated harness (`scripts/eval.py`, `scripts/bakeoff.py`) checks schema validity, triage policy, and citation honesty across all four languages, and the unit suite covers Unicode retrieval, topic RAG, ONNX inference, and the prompt/QA contracts.

## Honest limitations

The profiler numbers above are from a dev laptop; we have not yet frozen a run on the Ubuntu 22.04 / 8 GB target machine. The Yorùbá, Hausa, and Igbo passages are careful paraphrases still waiting on native clinician review (`data/eval/native_review_pending.json`), so the African-language bonus claim should be read as harness-supported, not clinician-verified. The vision model's specificity needs work on a larger, properly labelled dataset before anyone should lean on a negative screen. And `team_id` 1064863 should be checked against the ADTF portal one last time at submission.
