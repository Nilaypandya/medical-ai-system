"""
inference.py
============
Async inference orchestrator.

Ties together:
  1. preprocess.py  – image decoding + normalisation
  2. model_loader.py – retrieves the pre-loaded Keras model
  3. gradcam.py      – Grad-CAM heatmap generation

All heavy NumPy / TensorFlow work is executed in a thread-pool executor so
the FastAPI event loop is never blocked, allowing the server to handle
concurrent requests even on CPU-only deployments.
"""

from __future__ import annotations

import asyncio
from functools import partial

import numpy as np

from app.gradcam import generate_gradcam_overlay
from app.model_loader import get_class_labels, get_model
from app.preprocess import preprocess_for_gradcam


# ---------------------------------------------------------------------------
# Synchronous worker (runs inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------
def _infer_sync(image_bytes: bytes, disease_type: str) -> dict:
    """
    Blocking inference logic.  Executed off the main event loop.

    Steps:
      1. Preprocess the image into a batched tensor.
      2. Run model.predict() to get softmax probabilities.
      3. Pick the top-1 class.
      4. Compute Grad-CAM for the predicted class.
      5. Return a result dict.
    """
    # ── 1. Preprocess ────────────────────────────────────────────────────
    img_tensor, original_rgb = preprocess_for_gradcam(image_bytes)
    # img_tensor : (1, 224, 224, 3) float32
    # original_rgb : (224, 224, 3) uint8

    # ── 2. Model inference ───────────────────────────────────────────────
    model = get_model(disease_type)
    labels = get_class_labels(disease_type)

    # predict() returns shape (1, num_classes)
    probs = model.predict(img_tensor, verbose=0)[0]   # shape (num_classes,)

    # ── 3. Top-1 class ───────────────────────────────────────────────────
    class_index = int(np.argmax(probs))
    confidence = float(round(float(probs[class_index]), 4))
    prediction = labels[class_index]

    # Build a human-readable probability dict
    all_probs = {
        label: float(round(float(p), 4))
        for label, p in zip(labels, probs)
    }

    # ── 4. Grad-CAM heatmap ──────────────────────────────────────────────
    try:
        heatmap_b64 = generate_gradcam_overlay(
            model=model,
            img_tensor=img_tensor,
            original_rgb=original_rgb,
            class_index=class_index,
        )
    except Exception as exc:
        # Grad-CAM is best-effort; don't fail the prediction if it errors.
        print(f"⚠  Grad-CAM failed: {exc}")
        heatmap_b64 = ""

    # ── 5. Return ────────────────────────────────────────────────────────
    return {
        "prediction": prediction,
        "confidence": confidence,
        "heatmap": heatmap_b64,
        "all_probabilities": all_probs,
    }


# ---------------------------------------------------------------------------
# Public async wrapper
# ---------------------------------------------------------------------------
async def run_inference(image_bytes: bytes, disease_type: str) -> dict:
    """
    Async entry-point called by the FastAPI route handler.

    Offloads the CPU/GPU-bound work to a thread-pool so the event loop
    can continue handling other requests during inference.
    """
    loop = asyncio.get_running_loop()

    # `partial` is used to bind arguments since run_in_executor only accepts
    # a zero-argument callable.
    result = await loop.run_in_executor(
        None,                                      # use default ThreadPoolExecutor
        partial(_infer_sync, image_bytes, disease_type),
    )
    return result
