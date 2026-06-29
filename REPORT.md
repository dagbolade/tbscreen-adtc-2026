# TBScreenAI — Clinical TB Screening Assistant

**Team ID:** REPLACE_WITH_YOUR_TEAM_ID  
**Domain:** healthcare_medical  
**Model:** REPLACE_MODEL_NAME

---

## Problem

Tuberculosis kills 1.25 million people annually. Sub-Saharan Africa carries a disproportionate burden, and most rural clinics lack radiologists to interpret chest X-rays. Existing AI screening tools like Delft Imaging's CAD4TB and Qure.ai require cloud connectivity, cost upwards of $2,000 per unit, and produce outputs only in English.

Community health workers are the frontline of TB screening in rural Africa. They need clear, actionable guidance in the language their patients speak, not a probability score they cannot interpret.

We built an offline clinical assistant that screens chest X-rays using a trained computer vision model, then uses a local LLM to explain the result in plain language (English, Yoruba, Hausa, or Igbo), recommend triage actions, and educate patients. Everything runs on a standard laptop with no internet connection.

---

## Target User

Community health workers and primary health centre staff conducting TB screening in rural and peri-urban clinics across Nigeria and West Africa. These users typically have basic health training but are not radiologists. They need clear, actionable guidance, not medical jargon.

---

## Design Decisions

### Vision Model (Cross-Disciplinary Integration)

The chest X-ray screening model is MobileNetV3-Small, trained through 3-stage transfer learning on:

- TBX11K: 11,200 images (health / sick-but-non-TB / active TB)
- Shenzhen: 662 images (normal / TB-positive)
- OpenI: 7,056 frontal chest X-rays
- NIH ChestX-ray14: 112,120 images

Validation AUC: 0.930. Test AUC: 0.888. Production threshold t=0.65 (sensitivity 70.2%, FPR 11.7%).

The model runs as an ONNX file (4.2 MB) using ONNX Runtime with CPUExecutionProvider. Inference takes under 2 seconds on CPU. No TensorFlow dependency at runtime.

This model was not built for this competition. It was built as part of TBScreenAI, an offline-first TB screening tool being developed for deployment in Nigerian hospitals. The private development repository (dagbolade/TBScreenAI) contains the full training pipeline, evaluation code, and deployment infrastructure.

### LLM Selection

REPLACE: Which model, why, what alternatives were considered

### Quantization

REPLACE: Quantization level chosen, why, what tradeoffs

### Integration Architecture

The vision model outputs a TB probability score from the chest X-ray. This feeds into a structured prompt template that includes the probability, urgency level, and clinical context. The LLM generates three outputs:

1. Clinical interpretation for the health worker
2. Triage recommendation (referral urgency, next diagnostic steps)
3. Patient-facing explanation in the selected language

This is a load-bearing integration. The LLM cannot produce a clinically meaningful interpretation without the vision model's output. Remove the vision model and you have a generic medical chatbot. Remove the LLM and you have a probability score with no explanation. Together they form a complete screening-to-decision pipeline.

### African Language Support

Languages: English (en), Yoruba (yo), Hausa (ha), Igbo (ig).

REPLACE: Approach to language support, quality evaluation, any fine-tuning

---

## Constraints

- 8 GB RAM ceiling (7 GB effective budget for the application)
- No discrete GPU — CPU-only inference on Intel Core i5 / AMD Ryzen 5
- 100% offline operation — no cloud API calls, no internet dependency during inference
- Clinical accuracy matters — wrong triage recommendations have real consequences
- Language quality matters — mistranslation in a medical context is dangerous

---

## Benchmarks

All measurements on REPLACE_MACHINE_SPEC:

| Metric | Value |
|---|---|
| LLM tokens/sec | REPLACE |
| LLM peak RAM (RSS) | REPLACE |
| Vision model inference time | REPLACE |
| Vision model RAM overhead | ~15 MB |
| Total peak RAM | REPLACE |
| CPU temperature under load | REPLACE |
