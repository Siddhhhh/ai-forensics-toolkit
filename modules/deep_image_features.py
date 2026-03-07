import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ----------------------------
# Load Pretrained ResNet18
# ----------------------------

@st.cache_resource
def load_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

resnet = load_resnet()

# Remove final classification layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_deep_features(image_bgr):
    """
    Takes OpenCV BGR image
    Returns 512-dimensional feature vector
    """

    # Convert BGR → RGB
    image_rgb = image_bgr[:, :, ::-1]

    # Convert to PIL
    pil_image = Image.fromarray(image_rgb)

    # Apply transforms
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        features = resnet(input_tensor)

    # Flatten to 1D vector
    features = features.view(-1).numpy()

    return features