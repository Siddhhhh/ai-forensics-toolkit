import cv2
import numpy as np


def enhance_image(image_bytes):
    """
    Stable HD enhancement:
    - 2x Upscale
    - Mild denoise
    - Unsharp masking (safe)
    - Light contrast boost
    """

    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return None

    # 1️⃣ Upscale
    height, width = image.shape[:2]
    upscaled = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    # 2️⃣ Light denoise (very mild)
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 3, 3, 7, 21)

    # 3️⃣ Unsharp Mask (safer than custom kernel)
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.4, blur, -0.4, 0)

    # 4️⃣ Light contrast boost
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1, beta=5)

    # Convert BGR → RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    return enhanced_rgb