import random


# ==================================================
# LANGUAGE POOLS — OBSERVATIONS
# ==================================================

STRUCTURE_SUBJECTS = [
    "The sentence structure",
    "The overall phrasing",
    "The writing pattern",
    "The way ideas are expressed",
    "The composition of the text",
    "The structural flow of the content"
]

STRUCTURE_ISSUES = [
    "appears unusually uniform",
    "shows high consistency",
    "lacks natural variation",
    "follows a repetitive pattern",
    "maintains a predictable structure",
    "exhibits minimal stylistic deviation"
]

PREDICTABILITY_SUBJECTS = [
    "Word choice",
    "Phrase selection",
    "Sentence progression",
    "Linguistic flow",
    "Vocabulary usage"
]

PREDICTABILITY_ISSUES = [
    "seems highly predictable",
    "follows common probability patterns",
    "shows limited spontaneity",
    "appears statistically driven",
    "demonstrates low lexical diversity"
]


# ==================================================
# LANGUAGE POOLS — REASONING
# ==================================================

REASONING_FRAMES = [
    "Such patterns are commonly associated with",
    "These characteristics are often observed in",
    "This behavior frequently appears in",
    "These signals are typical of",
    "This writing style is consistent with",
    "This pattern is frequently produced by"
]

CAUTION_FRAMES = [
    "However, this alone is not definitive.",
    "On its own, this does not prove automation.",
    "This observation should be interpreted cautiously.",
    "Human writing can occasionally exhibit similar traits.",
    "Contextual factors may also influence these patterns."
]


# ==================================================
# VERDICT TIERS (8 DISTINCT VERDICTS)
# ==================================================

VERDICT_TIERS = {
    "very_high": [
        "Highly likely AI-generated content.",
        "Strong indicators of automated text generation.",
        "Multiple independent signals point toward AI authorship."
    ],
    "high": [
        "Probable AI involvement in text creation.",
        "Several AI-like characteristics were detected.",
        "The writing style aligns with automated generation patterns."
    ],
    "medium_high": [
        "Moderate indications of AI-assisted writing.",
        "The text exhibits noticeable algorithmic traits.",
        "Some features resemble AI-generated content."
    ],
    "medium": [
        "Mixed signals detected — the origin of the text is unclear.",
        "The text exhibits both human-like and AI-like traits.",
        "Insufficient evidence for definitive classification."
    ],
    "medium_low": [
        "Slight indicators of automation, but evidence is weak.",
        "The content shows minor structured patterns.",
        "AI involvement cannot be ruled out, but remains unlikely."
    ],
    "low": [
        "The text is likely written by a human.",
        "Writing patterns are consistent with natural human authorship.",
        "No strong indicators of AI-generated content were observed."
    ],
    "very_low": [
        "The writing strongly reflects human spontaneity.",
        "Linguistic variation aligns with natural human expression.",
        "The content appears authentically human-written."
    ]
}


# ==================================================
# CORE VERDICT GENERATOR
# ==================================================

def generate_text_verdict(risk, signals):
    observations = []
    reasoning = []

    # -------- OBSERVATIONS --------
    if signals.get("structure_uniformity", 0) > 0.6:
        observations.append(
            f"{random.choice(STRUCTURE_SUBJECTS)} "
            f"{random.choice(STRUCTURE_ISSUES)}."
        )

    if signals.get("predictability", 0) > 0.6:
        observations.append(
            f"{random.choice(PREDICTABILITY_SUBJECTS)} "
            f"{random.choice(PREDICTABILITY_ISSUES)}."
        )

    if not observations:
        observations.append(
            "No strong structural or linguistic irregularities were detected."
        )
# -------- REASONING --------
    ai_signals_present = (
        signals.get("structure_uniformity", 0) > 0.6 or
        signals.get("predictability", 0) > 0.6
    )

    if ai_signals_present:
        reasoning.append(
            f"{random.choice(REASONING_FRAMES)} "
            "AI-generated or algorithmically assisted text."
        )
    if random.random() < 0.6:
        reasoning.append(random.choice(CAUTION_FRAMES))
    else:
        reasoning.append(
            "The absence of strong structural or predictability signals suggests natural human writing patterns."
        )
    # -------- VERDICT SELECTION --------
    if risk > 0.9:
        tier = "very_high"
        probability = "Very high likelihood of AI involvement"
    elif risk > 0.75:
        tier = "high"
        probability = "High likelihood of AI involvement"
    elif risk > 0.6:
        tier = "medium_high"
        probability = "Moderate likelihood of AI involvement"
    elif risk > 0.45:
        tier = "medium"
        probability = "Uncertain — mixed evidence detected"
    elif risk > 0.3:
        tier = "medium_low"
        probability = "Low to moderate likelihood of AI involvement"
    elif risk > 0.15:
        tier = "low"
        probability = "Low likelihood of AI involvement"
    else:
        tier = "very_low"
        probability = "Very low likelihood of AI involvement"

    verdict = random.choice(VERDICT_TIERS[tier])

    return {
        "observations": observations,
        "reasoning": reasoning,
        "verdict": verdict,
        "probability": probability
    }
