"""
preprocess.py
=============
All image-preprocessing logic lives here so it can be tested independently
of the model and reused across inference and training pipelines.

Pipeline:
  raw bytes → PIL Image → resize → numpy array → normalise → batched tensor
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

# All models were trained with 224×224 RGB input
TARGET_SIZE: tuple[int, int] = (224, 224)


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """
    Decode raw image bytes (JPEG / PNG) into a PIL Image.

    Raises:
        ValueError: if the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc


def normalise_channels(img: Image.Image) -> Image.Image:
    """
    Ensure the image is RGB (3-channel).

    - RGBA / P (palette) images are converted to RGB.
    - Grayscale (L) images are replicated across 3 channels.
    - CMYK images are converted to RGB.

    Medical images (DICOM renders, greyscale X-rays, etc.) often arrive in
    non-standard channel configurations, so this step is essential.
    """
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGBA").convert("RGB")
    elif img.mode == "L":
        img = ImageOps.colorize(img, black="black", white="white")
        img = img.convert("RGB")
    elif img.mode == "CMYK":
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize(img: Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """
    Resize to the target spatial dimensions using high-quality Lanczos resampling.
    Aspect ratio is NOT preserved (centre-crop or pad in production if needed).
    """
    return img.resize(size, Image.Resampling.LANCZOS)


def pil_to_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL Image → float32 numpy array in [0, 1].

    Output shape: (H, W, 3)
    """
    arr = np.array(img, dtype=np.float32)
    # Scale pixel values from [0, 255] to [0.0, 1.0]
    arr /= 255.0
    return arr


def add_batch_dim(arr: np.ndarray) -> np.ndarray:
    """
    Add a leading batch dimension: (H, W, 3) → (1, H, W, 3).
    Keras / TF models expect a batch even for single-image inference.
    """
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Public API – single convenience function used by inference.py
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.

    Returns:
        Batched float32 tensor of shape (1, 224, 224, 3) ready for model.predict().
    """
    img = bytes_to_pil(image_bytes)          # decode
    img = normalise_channels(img)             # ensure RGB
    img = resize(img)                         # 224 × 224
    arr = pil_to_array(img)                   # → [0,1] float32
    tensor = add_batch_dim(arr)               # (1, H, W, 3)
    return tensor


def preprocess_for_gradcam(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns both the preprocessed tensor AND the original resized uint8 array
    (needed by the Grad-CAM overlay renderer).

    Returns:
        (tensor, original_uint8)  shapes: (1,224,224,3), (224,224,3)
    """
    img = bytes_to_pil(image_bytes)
    img = normalise_channels(img)
    img = resize(img)

    original_uint8 = np.array(img, dtype=np.uint8)   # keep for overlay
    tensor = add_batch_dim(pil_to_array(img))          # normalised for model

    return tensor, original_uint8
