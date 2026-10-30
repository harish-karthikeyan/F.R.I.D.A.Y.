
from PIL import Image


def load_food_image(uploaded_file):
    """
    Load and validate an uploaded food image.
    Returns a PIL image or raises ValueError.
    """
    if uploaded_file is None:
        raise ValueError("Please upload a food image.")

    try:
        image = Image.open(uploaded_file).convert("RGB")
        return image
    except Exception as exc:
        raise ValueError("Unable to read this image.") from exc


def recognize_food(image):
    """
    Placeholder for the future food-recognition model.
    Image loading works, but identification requires a trained model.
    """
    if image is None:
        raise ValueError("No image provided.")

    return {
        "recognized": False,
        "food_name": None,
        "confidence": None,
        "message": (
            "Image loaded successfully. A food-recognition model "
            "must be connected to identify the food."
        ),
    }
