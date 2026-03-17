"""
training/train_model.py
=======================
End-to-end training pipeline for medical image classification.

Supports three disease types:
  • brain_tumor   – MRI scans (4-class)
  • pneumonia     – Chest X-Rays (2-class)
  • skin_cancer   – Dermoscopy images (7-class HAM10000 subset)

Usage:
    python training/train_model.py --disease brain_tumor --data_dir /data/brain_tumor

Dataset directory layout expected:
    <data_dir>/
        train/
            class_a/  *.jpg / *.png
            class_b/
        val/
            class_a/
            class_b/

Public datasets:
  • Brain Tumor MRI: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
  • Chest X-Ray Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  • Skin Cancer HAM10000: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10          # frozen backbone warm-up
FINE_TUNE_EPOCHS = 10        # unfrozen fine-tuning
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5

CLASS_LABELS = {
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


# ---------------------------------------------------------------------------
# 1. Data loading with augmentation
# ---------------------------------------------------------------------------
def build_datasets(data_dir: str) -> tuple:
    """
    Load train / val image datasets with augmentation.

    Augmentation strategy (training only):
      - Random horizontal + vertical flip
      - Random rotation ±15°
      - Random zoom ±10%
      - Random brightness / contrast jitter
    These transforms improve generalisation on small medical datasets.
    """
    train_ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    # ── Augmentation layer (applied only during training) ─────────────────
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    # ── Normalise pixel values [0,255] → [0,1] ────────────────────────────
    normalise = layers.Rescaling(1.0 / 255)

    def augment_and_norm(x, y):
        x = augment(x, training=True)
        x = normalise(x)
        return x, y

    def norm_only(x, y):
        x = normalise(x)
        return x, y

    # Cache → augment/norm → prefetch for performance
    train_ds = (
        train_ds
        .cache()
        .map(augment_and_norm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .cache()
        .map(norm_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# 2. Model architecture
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    """
    Transfer learning with MobileNetV2.

    Returns:
        (model, backbone)  – backbone reference needed for fine-tuning.

    Two-phase training strategy:
      Phase 1 (frozen backbone): Train only the classifier head.
               The backbone already knows low-level features from ImageNet.
               This avoids destroying those features with a large LR.
      Phase 2 (unfrozen fine-tuning): Unfreeze the backbone and train
               end-to-end with a very small LR.
    """
    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False   # Phase 1: frozen

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MedicalCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    model.summary()
    return model, backbone


# ---------------------------------------------------------------------------
# 3. Callbacks
# ---------------------------------------------------------------------------
def build_callbacks(disease: str, output_dir: str) -> list:
    """
    Training callbacks:
      - ModelCheckpoint: save best val_accuracy checkpoint
      - EarlyStopping:   stop if val_loss doesn't improve for 5 epochs
      - ReduceLROnPlateau: halve LR on plateau
      - TensorBoard:     logs for TensorBoard visualisation
    """
    os.makedirs(output_dir, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f"{disease}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, "logs", disease),
            histogram_freq=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
def train(disease: str, data_dir: str, output_dir: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Training model for: {disease.upper()}")
    print(f"{'='*60}")

    num_classes = len(CLASS_LABELS[disease])
    train_ds, val_ds = build_datasets(data_dir)
    model, backbone = build_model(num_classes)

    # ── Phase 1: warm-up (frozen backbone) ────────────────────────────────
    print("\n▶ Phase 1 – classifier head warm-up")
    hist1 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=build_callbacks(disease, output_dir),
        verbose=1,
    )

    # ── Phase 2: fine-tuning (unfreeze top 40 backbone layers) ────────────
    print("\n▶ Phase 2 – fine-tuning (top backbone layers unfrozen)")
    backbone.trainable = True
    # Only unfreeze the top layers; early layers detect generic edges/textures
    for layer in backbone.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    hist2 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=build_callbacks(disease, output_dir),
        verbose=1,
    )

    # ── Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(output_dir, f"{disease}_model.keras")
    model.save(final_path)
    print(f"\n✅ Model saved → {final_path}")

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluate(model, val_ds, disease, hist1, hist2, output_dir)


# ---------------------------------------------------------------------------
# 5. Evaluation & plots
# ---------------------------------------------------------------------------
def evaluate(model, val_ds, disease, hist1, hist2, output_dir):
    print("\n▶ Evaluation on validation set")
    loss, acc, auc = model.evaluate(val_ds, verbose=0)
    print(f"  Val Loss:     {loss:.4f}")
    print(f"  Val Accuracy: {acc:.4f}")
    print(f"  Val AUC:      {auc:.4f}")

    # Confusion matrix + classification report
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    labels = CLASS_LABELS[disease]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # ── Plot training curves ───────────────────────────────────────────────
    all_acc = hist1.history["accuracy"] + hist2.history["accuracy"]
    all_val_acc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    all_loss = hist1.history["loss"] + hist2.history["loss"]
    all_val_loss = hist1.history["val_loss"] + hist2.history["val_loss"]

    epochs_range = range(len(all_acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, all_acc, label="Train Accuracy")
    axes[0].plot(epochs_range, all_val_acc, label="Val Accuracy")
    axes[0].axvline(INITIAL_EPOCHS, color="red", linestyle="--", label="Fine-tune start")
    axes[0].set_title(f"{disease} – Accuracy")
    axes[0].legend()

    axes[1].plot(epochs_range, all_loss, label="Train Loss")
    axes[1].plot(epochs_range, all_val_loss, label="Val Loss")
    axes[1].axvline(INITIAL_EPOCHS, color="red", linestyle="--", label="Fine-tune start")
    axes[1].set_title(f"{disease} – Loss")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{disease}_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"  Training curves saved → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Medical AI model")
    p.add_argument(
        "--disease",
        required=True,
        choices=list(CLASS_LABELS.keys()),
        help="Which disease model to train",
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Root directory containing train/ and val/ sub-folders",
    )
    p.add_argument(
        "--output_dir",
        default="model",
        help="Directory to save trained model and plots (default: model/)",
    )
    return p.parse_args()


if __name__ == "__main__": 
    # Optional: configure GPU memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"🖥  GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("💻 No GPU found – training on CPU (will be slow)")

    args = parse_args()
    train(args.disease, args.data_dir, args.output_dir)
