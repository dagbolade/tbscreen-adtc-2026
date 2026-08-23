# TBScreen

Offline AI-assisted chest X-ray TB screening with WHO-guideline Q&A in English, Yorùbá, Hausa, and Igbo. A health worker uploads a CXR, gets a TB probability, a deterministic WHO-aligned triage decision, a cited plain-language interpretation, and a clinical Q&A chat, all on an 8 GB laptop with zero network access.

**ADTC 2026 Laptop LLM track · Team 1064863 · healthcare_medical**
Public repo: https://github.com/dagbolade/tbscreen-adtc-2026

---

## Submission facts

`metadata.json` is the source of truth; its final values:

| Field                          | Value                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `team_id`                    | `1064863`                                                                                                                                            |
| `domain`                     | `healthcare_medical`                                                                                                                                 |
| `language_scope`             | `en`, `yo`, `ha`, `ig`                                                                                                                         |
| `african_alpha_claim`        | `true` (four-language UI + localized corpus)                                                                                                         |
| `budget_laptop_claim`        | `true` (peak RSS ~3.1-3.3 GB vs 7 GB budget)                                                                                                         |
| `submitter`                  | David Agbolade · dagbolade72@gmail.com · github.com/dagbolade                                                                                        |
| `cross_disciplinary_pairing` | `computer_vision`, load-bearing (MobileNetV3 ONNX screener feeds the LLM pipeline)                                                                   |
| `test_prompts`               | exactly 2:`tp_001` (explain a 78.3% CXR screen for a symptomatic 34-year-old) and `tp_002` (positive screen vs confirmed diagnosis at a rural PHC) |
| `model`                      | `gemma-4-E2B-it-Q4_K_M`, runtime `llama.cpp`, quantization `GGUF Q4_K_M`, ~2.6B effective, packaging `binary_bundle`                           |
| `_runtime.model_path`        | `model/gemma-4-E2B-it-Q4_K_M.gguf` (downloaded by `download_model.sh`, never committed)                                                            |

## Checklist status

- [X] Repository public on GitHub
- [X] `metadata.json` fully filled, no placeholders, exactly 2 test prompts
- [X] `download_model.sh` idempotent, credential-free, pinned SHA-256 `9378bc…8672d`, public Hugging Face URL
- [X] Valid GGUF weights; `model/` and `*.gguf` gitignored
- [X] `REPORT.md` technical writeup with measured numbers
- [X] 100% offline inference: local ONNX + GGUF + TF-IDF index, system fonts only, no CDN
- [X] Offline TF-IDF index committed at `corpus/index/`
- [X] Clinical Q&A path (`TBScreenAssistant.ask` / `POST /ask`) aligned with the test prompts
- [X] Linux validation: Ubuntu 22.04 container at 4 CPUs / 8 GB, 19/19 unit tests, full e2e pass, peak RSS 3.33 GB (details in REPORT.md)
- [X] Participant profiler run recorded (16.73 tok/s, RSS 2135 MB; Sperf 100.0, Seff 69.5; see REPORT.md)
- [ ] Official-laptop (Ubuntu 22.04 / i5 / 8 GB) profiler numbers: measured by the audit sandbox
- [ ] 2-minute demo video: submitted on DevPost

## Repository layout

```
tbscreen-adtc-2026/
├── metadata.json              Team/model/prompt metadata (final)
├── download_model.sh          Downloads the GGUF to model/ (SHA-pinned)
├── REPORT.md                  Technical writeup
├── requirements.txt           Pinned deps (Python 3.11+ required on Linux)
├── src/tbscreen/              Flask app, pipeline, RAG, prompts, sanitizers
├── vision/                    MobileNetV3-Small ONNX screener + weights
├── corpus/                    98 WHO-guideline passages (en/yo/ha/ig) + TF-IDF index
├── samples/                   20 Shenzhen CXRs for smoke tests
├── scripts/                   build_rag_index, eval, bakeoff
├── tests/                     contracts, retrieval policy, e2e pipeline
└── model/                     GGUF target dir (gitignored, populated by script)
```

---

## Run instructions

```bash
python -m venv .venv && source .venv/bin/activate
# Linux (e.g. Ubuntu 22.04): sudo apt install -y build-essential cmake
# (llama-cpp-python ships no manylinux wheels and compiles from source)
# Ubuntu 22.04's system Python 3.10 has no wheels for the pinned numpy/onnxruntime:
# install Python 3.12 first (e.g. from the deadsnakes PPA), then create the venv with it
pip install -r requirements.txt
bash download_model.sh
python scripts/build_rag_index.py   # after corpus edits
python -m tbscreen.app              # from repo root after: export PYTHONPATH=src:.
# or:
PYTHONPATH=src:. python src/tbscreen/app.py
# UI: http://127.0.0.1:5000  (CXR screen + Clinical Q&A)

python scripts/eval.py
python scripts/bakeoff.py --candidate gemma4-e2b-q4km
python tests/test_e2e_pipeline.py
python -m unittest tests.test_retrieval_policy tests.test_contracts
```

Gate 1 deadline: **August 25, 2026**. Official profiler target: Ubuntu 22.04 / 8 GB ADTC Standard Laptop.

## Local profiler (participant mode)

```bash
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
# The profiler needs llama-bench on PATH:
#   macOS: brew install llama.cpp    Linux: build llama.cpp (the source is vendored
#   inside the llama-cpp-python sdist, so the exact runtime version can be rebuilt)
bash download_model.sh
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json \
  --skip-accuracy
```

A valid run produces `submission.json` with `"measured_on": "participant_laptop"`. Our recorded run: 16.73 tok/s generation, 2135 MB peak RSS, no throttling (Apple M1 Pro participant machine; REPORT.md explains what does and does not transfer to the target laptop).

## Rules honored

Public repo, no weights in git (the evaluator downloads weights via the script), 100% offline during evaluation, llama.cpp-only GGUF runtime, within the 8 GB RAM profile (4 vCPU, integrated GPU), exactly 2 test prompts. Full rules and eligibility: [adtc-2026.devpost.com/rules](https://adtc-2026.devpost.com/rules). Support: [challenge@africadeeptech.org](mailto:challenge@africadeeptech.org).

---

## License

This project is licensed under the terms of the [GNU GPL v3 License](LICENSE).
