import streamlit as st
import cv2
import numpy as np
import joblib

from modules.deep_image_features import extract_deep_features


MODEL_PATH = "models/image_ai_detector.pkl"

@st.cache_resource
def load_image_model():
    return joblib.load(MODEL_PATH)

model = load_image_model()

def check_image_authenticity(uploaded_file):

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "ai_probability": 0.0,
            "verdict": "Unclear"
        }

    # Extract deep 512-dim features
    features = extract_deep_features(image)

    feature_vector = np.array([features])

    # Predict probability
    ai_probability = model.predict_proba(feature_vector)[0][1]

    # Conservative thresholds
    if ai_probability > 0.85:
        verdict = "Likely AI-generated image"
    elif ai_probability < 0.25:
        verdict = "Likely camera-captured image"
    else:
        verdict = "Uncertain — mixed signals detected"

    return {
        "ai_probability": round(ai_probability * 100, 1),
        "verdict": verdict
    }