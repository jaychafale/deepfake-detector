import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize to target and normalize to [0,1], return (1, H, W, C)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def validate_image(image: Image.Image) -> bool:
    """Basic integrity/size checks for PIL images."""
    try:
        w, h = image.size
        return (w > 0 and h > 0)
    except Exception:
        return False
