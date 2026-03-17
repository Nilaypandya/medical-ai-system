"""
model_loader.py
===============
Responsible for loading (or generating demo) Keras models at startup.

In production:
  - Replace `_build_demo_model()` with `tf.keras.models.load_model(path)`
  - Place your .keras / .h5 files under the `model/` directory.

The module exposes a single dict `MODELS` that is populated once and
then re-used by every inference request without re-loading from disk.
"""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Global model registry  {disease_type: keras.Model}
# ---------------------------------------------------------------------------
MODELS: dict[str, keras.Model] = {}

# ---------------------------------------------------------------------------
# Class labels per disease
# ---------------------------------------------------------------------------
CLASS_LABELS: dict[str, list[str]] = {
    "brain_tumor": ["glioma", "meningioma", "no_tumor", "pituitary"],
    "pneumonia": ["normal", "pneumonia"],
    "skin_cancer": [
        "actinic_keratosis",
        "basal_cell_carcinoma",
        "benign_keratosis",
        "dermatofibroma",
        "melanocytic_nevi",
        "melanoma",
        "vascular_lesion",
    ],
}

# Image size expected by all models (can be per-model in production)
IMG_SIZE = (224, 224)

# Path where real saved models live
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def _build_demo_model(num_classes: int) -> keras.Model:
    """
    Builds a lightweight MobileNetV2-based transfer-learning model.

    Architecture:
      1. MobileNetV2 backbone (ImageNet weights, frozen)
      2. Global Average Pooling
      3. Dropout (regularisation)
      4. Dense softmax head

    In a real training pipeline the backbone would be unfrozen after
    initial warm-up and fine-tuned end-to-end.
    """
    # ── Backbone ────────────────────────────────────────────────────────────
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,          # drop the ImageNet classifier
        weights="imagenet",
    )
    backbone.trainable = False      # freeze during initial training

    # ── Classification head ─────────────────────────────────────────────────
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MedicalCNN")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _load_or_build(disease: str) -> keras.Model:
    """
    Try to load a saved model from disk; fall back to building a demo model.

    File naming convention:  model/<disease>_model.keras
    """
    saved_path = os.path.join(MODEL_DIR, f"{disease}_model.keras")
    if os.path.exists(saved_path):
        print(f"  ✔ Loading saved model: {saved_path}")
        return keras.models.load_model(saved_path)
    else:
        print(f"  ⚠  No saved model found for '{disease}'. Building demo model …")
        return _build_demo_model(num_classes=len(CLASS_LABELS[disease]))


def load_all_models() -> None:
    """
    Called once at application startup (see lifespan in main.py).
    Populates the global MODELS dict so inference calls don't pay
    the model-load cost on every request.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    for disease in CLASS_LABELS:
        print(f"Loading model for: {disease}")
        MODELS[disease] = _load_or_build(disease)

    print(f"✅ {len(MODELS)} models loaded into memory")


def get_model(disease: str) -> keras.Model:
    """Thread-safe read of the already-loaded model."""
    if disease not in MODELS:
        raise KeyError(f"Model for '{disease}' not loaded. Call load_all_models() first.")
    return MODELS[disease]


def get_class_labels(disease: str) -> list[str]:
    return CLASS_LABELS[disease]
