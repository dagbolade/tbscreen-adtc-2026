"""
integration/pipeline.py — Connects the vision model output to the LLM.

This is the load-bearing integration layer. The vision model produces a
TB probability score, and this module formats it into a structured prompt
that the LLM uses to generate clinical interpretations, triage recommendations,
and patient education in the selected language.

The LLM cannot produce a clinically meaningful response without this pipeline.
Remove the vision model and you have a generic chatbot. Remove the LLM and you
have a probability score with no explanation. Together they form a complete
screening-to-decision system.
"""

from pathlib import Path


# Supported languages with BCP-47 codes
SUPPORTED_LANGUAGES = {
    "en": "English",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
}


def build_screening_prompt(vision_result: dict, language: str = "en") -> str:
    """
    Build a structured prompt for the LLM from the vision model's screening output.

    Parameters
    ----------
    vision_result : dict
        Output from TBScreenModel.predict(), containing:
        - tb_probability (float)
        - prediction (str)
        - screening_result (str)
        - threshold (float)
    language : str
        BCP-47 language code: "en", "yo", "ha", or "ig".

    Returns
    -------
    str
        A structured prompt ready for the LLM.
    """
    tb_prob = vision_result["tb_probability"]
    result = vision_result["screening_result"]
    threshold = vision_result["threshold"]

    lang_name = SUPPORTED_LANGUAGES.get(language, "English")

    # Determine clinical urgency level
    if tb_prob >= 0.85:
        urgency = "HIGH"
        urgency_context = "This is a high-probability result requiring urgent clinical attention."
    elif tb_prob >= threshold:
        urgency = "MODERATE"
        urgency_context = "This is a positive screening result that warrants clinical follow-up."
    elif tb_prob >= 0.40:
        urgency = "BORDERLINE"
        urgency_context = "This result is below the screening threshold but not definitively negative. Consider clinical context."
    else:
        urgency = "LOW"
        urgency_context = "This is a negative screening result with low TB probability."

    prompt = f"""You are a clinical decision support assistant for TB screening in rural African healthcare settings. You are speaking to a community health worker who may not have advanced medical training.

SCREENING RESULT:
- TB Probability: {tb_prob * 100:.1f}%
- Screening Result: {result}
- Urgency Level: {urgency}
- Clinical Context: {urgency_context}
- Screening Threshold: {threshold * 100:.0f}%

IMPORTANT: This is an AI screening result from a chest X-ray analysis, NOT a diagnosis. All positive or borderline results must be confirmed by sputum testing (GeneXpert or smear microscopy) and reviewed by a clinician.

Respond in {lang_name}. Provide exactly three sections:

1. INTERPRETATION: Explain what this screening result means in simple, clear language that a community health worker can understand. Do not use complex medical jargon.

2. RECOMMENDED ACTION: Give specific, actionable next steps based on the urgency level:
   - HIGH: Immediate referral to the nearest facility with GeneXpert. Isolate patient if possible. Start contact tracing.
   - MODERATE: Refer for sputum testing within 48 hours. Document symptoms (cough duration, night sweats, weight loss).
   - BORDERLINE: Assess symptoms. If symptomatic (cough > 2 weeks, night sweats, weight loss), refer for testing. If asymptomatic, rescreen in 2 weeks.
   - LOW: No immediate action needed. Advise patient to return if symptoms develop.

3. PATIENT EXPLANATION: Write a brief, reassuring explanation that the health worker can read directly to the patient in {lang_name}. Keep it simple and culturally appropriate. Avoid causing unnecessary panic for positive results while emphasizing the importance of follow-up."""

    return prompt


def build_clinical_qa_prompt(question: str, language: str = "en") -> str:
    """
    Build a prompt for general clinical Q&A about TB.

    This mode does not use vision model output — it's pure clinical knowledge.
    Used when health workers have general questions about TB management.

    Parameters
    ----------
    question : str
        The clinical question from the health worker.
    language : str
        BCP-47 language code.

    Returns
    -------
    str
        A structured prompt for the LLM.
    """
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")

    prompt = f"""You are a clinical decision support assistant specializing in tuberculosis (TB) for community health workers in rural Africa. Answer questions accurately and practically.

GUIDELINES:
- Use simple, clear language appropriate for health workers without advanced medical training
- Give practical, actionable advice suited to resource-limited settings
- When recommending diagnostic steps, prioritize what is available in primary health centres (symptom screening, sputum collection, referral pathways)
- Always note when a question requires a qualified clinician's judgment
- Never provide a definitive diagnosis — you are a decision support tool

Respond in {lang_name}.

QUESTION: {question}"""

    return prompt


def run_screening_pipeline(
    image_path: str,
    language: str = "en",
    vision_model=None,
    llm=None,
):
    """
    Full screening pipeline: X-ray image in, clinical interpretation out.

    Parameters
    ----------
    image_path : str
        Path to the chest X-ray image.
    language : str
        Output language code.
    vision_model : TBScreenModel
        The vision model instance.
    llm : callable
        LLM inference function. Takes a prompt string, returns a response string.

    Returns
    -------
    dict with keys:
        vision_result : dict (raw screening output)
        prompt        : str (the prompt sent to the LLM)
        interpretation: str (LLM's clinical interpretation)
        language      : str (language used)
    """
    # Step 1: Vision model screens the X-ray
    vision_result = vision_model.predict(image_path)

    # Step 2: Build structured prompt from vision output
    prompt = build_screening_prompt(vision_result, language)

    # Step 3: LLM generates clinical interpretation
    interpretation = llm(prompt)

    return {
        "vision_result": vision_result,
        "prompt": prompt,
        "interpretation": interpretation,
        "language": language,
    }
