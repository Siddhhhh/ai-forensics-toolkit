from rembg import remove
from PIL import Image
import io


def remove_background(uploaded_file):
    """
    Removes the background from an uploaded image.
    Returns a PIL Image with transparent background.
    """

    # Read image bytes
    image_bytes = uploaded_file.read()

    # Remove background
    output_bytes = remove(image_bytes)

    # Convert back to PIL Image
    output_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    return output_image