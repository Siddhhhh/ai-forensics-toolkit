print("TEXT TRAINING STARTED")

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# -----------------------------
# SETTINGS
# -----------------------------
SAMPLE_SIZE = 3000  # per class
MAX_LENGTH = 256

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading CSV files...")
fake_df = pd.read_csv("datasets/Fake.csv")
true_df = pd.read_csv("datasets/True.csv")

# Downsample
fake_df = fake_df.sample(SAMPLE_SIZE, random_state=42)
true_df = true_df.sample(SAMPLE_SIZE, random_state=42)

fake_df["label"] = 1
true_df["label"] = 0

df = pd.concat([fake_df, true_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

texts = df["text"].astype(str).tolist()
labels = df["label"].values

print(f"Total samples used: {len(texts)}")

# -----------------------------
# LOAD DISTILBERT
# -----------------------------
print("Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

# -----------------------------
# EXTRACT EMBEDDINGS
# -----------------------------
def get_embedding(text):
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


print("Extracting embeddings (this will take time)...")
embeddings = []

for i, text in enumerate(texts):
    emb = get_embedding(text)
    embeddings.append(emb)

    if i % 500 == 0:
        print(f"Processed {i} texts...")

X = np.array(embeddings)
y = labels

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN RANDOM FOREST
# -----------------------------
print("Training RandomForest...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

clf.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/text_ai_detector_v2.pkl")

print("Model saved to models/text_ai_detector_v2.pkl")