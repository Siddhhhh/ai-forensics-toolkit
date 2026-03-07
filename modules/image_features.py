import cv2
import numpy as np


def extract_image_features(image):
    """
    Extract low-level forensic features from an image.
    Returns numerical signals used for AI image detection.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # 1. Noise estimation
    # -----------------------------
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

    # -----------------------------
    # 2. Texture variance
    # -----------------------------
    texture_variance = np.std(gray)

    # -----------------------------
    # 3. Frequency domain energy
    # -----------------------------
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    frequency_energy = np.mean(magnitude_spectrum)

    return {
        "noise_level": float(noise_level),
        "texture_variance": float(texture_variance),
        "frequency_energy": float(frequency_energy)
    }
