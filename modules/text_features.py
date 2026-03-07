import re
import numpy as np


def extract_text_features(text: str):
    text = text.strip()

    # -----------------------------
    # Sentence-level features
    # -----------------------------
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = [len(s.split()) for s in sentences]

    if len(sentence_lengths) > 1:
        avg_sentence_length = float(np.mean(sentence_lengths))
        sentence_variance = float(np.var(sentence_lengths))
    else:
        avg_sentence_length = 0.0
        sentence_variance = 0.0

    # -----------------------------
    # Word-level features
    # -----------------------------
    words = re.findall(r"\b\w+\b", text.lower())

    if words:
        lexical_diversity = len(set(words)) / len(words)
    else:
        lexical_diversity = 0.0

    # -----------------------------
    # Structural uniformity
    # -----------------------------
    structure_uniformity = 1.0 / (1.0 + sentence_variance)

    # -----------------------------
    # Predictability (heuristic)
    # -----------------------------
    common_words = {"the", "is", "and", "of", "to", "in", "that", "it", "for"}
    common_count = sum(1 for w in words if w in common_words)

    predictability = (
        common_count / len(words)
        if len(words) > 0
        else 0.0
    )

    return {
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "structure_uniformity": structure_uniformity,
        "predictability": predictability
    }
