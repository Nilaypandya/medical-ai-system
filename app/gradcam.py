"""
gradcam.py
==========
Gradient-weighted Class Activation Mapping (Grad-CAM).

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).

How it works:
  1. Forward pass the image through the model.
  2. Identify the last convolutional layer (richest spatial features).
  3. Compute the gradient of the target class score w.r.t. that layer's
     feature maps using a GradientTape.
  4. Pool the gradients spatially (global average pooling) → weights αₖ.
  5. Weight each feature map by αₖ and sum → raw CAM.
  6. Apply ReLU (keep only positive influence).
  7. Upscale the CAM to the original image size.
  8. Render as a colour heatmap and blend with the original image.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


# ---------------------------------------------------------------------------
# Helper: find the last Conv2D layer in the model
# ---------------------------------------------------------------------------
def _last_conv_layer(model: keras.Model) -> str:
    """
    Traverse model layers in reverse and return the name of the last
    Conv2D layer.  Works for Sequential, Functional, and sub-classed models.
    """
    for layer in reversed(model.layers):
        # Check the layer itself
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        # Check inside nested sub-models (e.g. MobileNetV2 backbone)
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found in model – cannot compute Grad-CAM.")


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------
def compute_gradcam(
    model: keras.Model,
    img_tensor: np.ndarray,   # shape (1, H, W, 3)
    class_index: int,
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for the given class index.

    Returns:
        heatmap: float32 array of shape (H, W) with values in [0, 1].
    """
    # ── Identify the target convolutional layer ──────────────────────────
    if conv_layer_name is None:
        conv_layer_name = _last_conv_layer(model)

    # Build a sub-model that outputs:
    #   (feature_maps_from_conv, final_predictions)
    # We need both to compute gradients.
    try:
        conv_layer = model.get_layer(conv_layer_name)
    except ValueError:
        # The layer might be inside a sub-model
        for layer in model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    conv_layer = layer.get_layer(conv_layer_name)
                    break
                except ValueError:
                    continue

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    # ── Forward pass + gradient computation ─────────────────────────────
    with tf.GradientTape() as tape:
        img_tf = tf.cast(img_tensor, tf.float32)
        conv_outputs, predictions = grad_model(img_tf)
        # Score for the target class (before softmax would give sharper maps,
        # but after softmax works well for explanation purposes)
        loss = predictions[:, class_index]

    # Gradient of the class score w.r.t. the conv feature maps
    grads = tape.gradient(loss, conv_outputs)   # (1, h, w, C)

    # ── Pool gradients spatially (αₖ in the paper) ───────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # shape (C,)

    # ── Weight feature maps and sum ──────────────────────────────────────
    conv_outputs = conv_outputs[0]                          # (h, w, C)
    # Multiply each channel by its weight
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]      # (h, w, 1)
    cam = tf.squeeze(cam)                                    # (h, w)

    # ── ReLU – keep only positive contributions ───────────────────────────
    cam = tf.nn.relu(cam).numpy()                           # (h, w)

    # ── Normalise to [0, 1] ───────────────────────────────────────────────
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def heatmap_to_rgb(cam: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Convert a float32 Grad-CAM array to a coloured (JET) RGB heatmap,
    resized to `target_size` = (width, height).

    Returns uint8 RGB array of shape (height, width, 3).
    """
    # Upscale to original image dimensions
    cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert [0,1] float → uint8 for OpenCV colormap
    cam_uint8 = np.uint8(255 * cam_resized)

    # Apply JET colormap: cold (blue) → warm (red) = low → high activation
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Convert BGR → RGB for PIL / web display
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    original_rgb: np.ndarray,    # uint8 (H, W, 3)
    heatmap_rgb: np.ndarray,     # uint8 (H, W, 3)
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Alpha-blend heatmap over the original image.

    alpha controls heatmap opacity:
      0.0 → only original image
      1.0 → only heatmap
    """
    blended = cv2.addWeighted(
        original_rgb.astype(np.float32),
        1.0 - alpha,
        heatmap_rgb.astype(np.float32),
        alpha,
        0,
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def array_to_base64_png(arr: np.ndarray) -> str:
    """
    Encode a uint8 RGB numpy array as a base-64 PNG string.
    Suitable for embedding directly in JSON API responses.
    """
    img = Image.fromarray(arr.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_gradcam_overlay(
    model: keras.Model,
    img_tensor: np.ndarray,     # (1, H, W, 3) float32
    original_rgb: np.ndarray,   # (H, W, 3) uint8
    class_index: int,
) -> str:
    """
    Full pipeline: compute Grad-CAM → render heatmap → overlay → base64 PNG.

    Returns:
        Base-64 encoded PNG string of the blended image.
    """
    h, w = original_rgb.shape[:2]

    # 1. Compute raw Grad-CAM
    cam = compute_gradcam(model, img_tensor, class_index)

    # 2. Resize to original image dimensions and colourise
    heatmap = heatmap_to_rgb(cam, target_size=(w, h))

    # 3. Blend with original image
    blended = overlay_heatmap(original_rgb, heatmap, alpha=0.45)

    # 4. Encode to base-64 PNG
    return array_to_base64_png(blended)
