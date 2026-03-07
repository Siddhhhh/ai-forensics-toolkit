print("SCRIPT STARTED")

import os
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from modules.deep_image_features import extract_deep_features


REAL_PATH = "dataset/real"
AI_PATH = "dataset/ai"


def load_images_from_folder(folder, label):
    data = []
    labels = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        image = cv2.imread(filepath)

        if image is None:
            continue

        features = extract_deep_features(image)

        data.append(features)
        labels.append(label)

    return data, labels


print("Loading real images...")
real_data, real_labels = load_images_from_folder(REAL_PATH, 0)

print("Loading AI images...")
ai_data, ai_labels = load_images_from_folder(AI_PATH, 1)

X = real_data + ai_data
y = real_labels + ai_labels

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/image_ai_detector.pkl")

print("\nDeep feature model saved to models/image_ai_detector.pkl")