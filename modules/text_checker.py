import streamlit as st
import joblib
import numpy as np

from modules.deep_text_features import extract_text_embedding

MODEL_PATH = "models/text_ai_detector_v2.pkl"
@st.cache_resource
def load_classifier():
    return joblib.load(MODEL_PATH)

clf = load_classifier()


def check_text_authenticity(text: str):

    text = text.strip()

    if len(text.split()) < 30:
        return {
            "risk": 0.2,
            "verdict": "Text too short for reliable analysis"
        }

    embedding = extract_text_embedding(text)
    embedding = np.array([embedding])

    ai_probability = clf.predict_proba(embedding)[0][1]

    if ai_probability > 0.85:
        verdict = "Likely AI / machine-generated text"
    elif ai_probability < 0.25:
        verdict = "Likely human-written text"
    else:
        verdict = "Uncertain — mixed linguistic signals"

    return {
        "risk": round(float(ai_probability) * 100, 1),
        "verdict": verdict
    }