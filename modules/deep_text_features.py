import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# LOAD DISTILBERT (same as training)
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

import streamlit as st

@st.cache_resource
def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    return tokenizer, model

tokenizer, model = load_text_model()

def extract_text_embedding(text: str):
    """
    Extract 768-dim CLS embedding from DistilBERT
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy().flatten()