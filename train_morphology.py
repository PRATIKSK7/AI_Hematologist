"""
AI Hematologist - Morphology Classification Training Script
===========================================================
This script trains an EfficientNetB0 model to classify 11 blood cell morphology types:
Normal, Macrocyte, Microcyte, Spherocyte, TargetCell, Stomatocyte, Ovalocyte, 
Schistocyte, BurrCell, Hypochromia, Elliptocyte.

Features:
- Transfer Learning & Fine-Tuning
- Mixed Precision (Apple Silicon compatible)
- Class Weight Balancing
- Data Augmentation & tf.data Pipelines
- Early Stopping, LR Plateau, Model Checkpointing
- Test Evaluation & Metric Visualizations
"""

import os
import sys
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

# Dataset Paths
TRAIN_DIR = "datasets/morphology_split/train"
VAL_DIR = "datasets/morphology_split/val"
TEST_DIR = "datasets/morphology_split/test"

# Output Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "morphology_model.keras")
LOG_DIR = "logs"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 10
SEED = 42

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Validate dataset presence
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Training directory not found at {TRAIN_DIR}")
    sys.exit(1)


# ==========================================================
# 2. HARDWARE SETUP (APPLE SILICON & MIXED PRECISION)
# ==========================================================

print("\n--- System Configuration ---")
try:
    gpus = tf.config.list_physical_devices()
    print(f"Available Devices: {gpus}")

    # Apple Silicon typically uses Metal Performance Shaders (MPS), handled natively by TF-Metal.
    # We attempt mixed precision, but fall back gracefully if unsupported.
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed Precision Enabled (mixed_float16)")
except Exception as e:
    print(f"Mixed Precision Disabled (not supported or error): {e}")


# ==========================================================
# 3. DATA PIPELINES
# ==========================================================

print("\n--- Loading Datasets ---")

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"\nDetected {num_classes} Classes:")
print(class_names)

# ==========================================================
# 4. CLASS WEIGHT COMPUTATION
# ==========================================================
# Address class imbalance by weighting rare classes heavier

print("\n--- Computing Class Weights ---")
labels = []
for _, y in train_ds.unbatch():
    labels.append(np.argmax(y.numpy()))

labels = np.array(labels)
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels,
)

class_weights = {i: float(w) for i, w in enumerate(weights)}
print("Calculated Class Weights:")
for i, w in class_weights.items():
    print(f"  {class_names[i]}: {w:.3f}")


# ==========================================================
# 5. PERFORMANCE OPTIMIZATION & AUGMENTATION
# ==========================================================

AUTOTUNE = tf.data.AUTOTUNE

# Cache and prefetch to eliminate I/O bottlenecks
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Data Augmentation layer (runs on GPU if available)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15),
], name="data_augmentation")


# ==========================================================
# 6. MODEL ARCHITECTURE
# ==========================================================

print("\n--- Building EfficientNetB0 Model ---")

# Load pre-trained base model
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

# Freeze base model layers initially
base_model.trainable = False

# Construct the model topology
inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x) # EfficientNet specific preprocessing
x = base_model(x, training=False) # Ensure batchnorm stays in inference mode
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

# Force dtype float32 for final softmax to maintain numerical stability in mixed precision
outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ==========================================================
# 7. CALLBACKS
# ==========================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
    ),
]


# ==========================================================
# 8. STAGE 1: FEATURE EXTRACTION
# ==========================================================

print("\n--- Stage 1: Training Top Layers ---")
try:
    history_feature_extract = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )
except KeyboardInterrupt:
    print("\nTraining interrupted manually.")


# ==========================================================
# 9. STAGE 2: FINE-TUNING
# ==========================================================

print("\n--- Stage 2: Fine-Tuning Base Model ---")

# Unfreeze the base model
base_model.trainable = True

# Refreeze bottom layers, fine-tune only top 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

try:
    history_fine_tune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )
except KeyboardInterrupt:
    print("\nFine-tuning interrupted manually.")


# ==========================================================
# 10. EVALUATION & METRICS
# ==========================================================

print("\n--- Evaluating on Test Set ---")

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n=> Test Accuracy: {test_acc:.4f}")

# Generate Predictions
print("Generating predictions...")
y_true = []
y_pred = []

for images, labels_batch in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels_batch.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate Classification Metrics
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n" + "="*40)
print("             TEST METRICS")
print("="*40)
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("="*40)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


# ==========================================================
# 11. VISUALIZATIONS
# ==========================================================

print("\n--- Generating Visualizations ---")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix - Morphology Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Saved: confusion_matrix.png")

# Extract history correctly depending on what stages executed
if 'history_fine_tune' in locals():
    # Append histories
    acc = history_feature_extract.history['accuracy'] + history_fine_tune.history['accuracy']
    val_acc = history_feature_extract.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
    loss = history_feature_extract.history['loss'] + history_fine_tune.history['loss']
    val_loss = history_feature_extract.history['val_loss'] + history_fine_tune.history['val_loss']
else:
    acc = history_feature_extract.history['accuracy']
    val_acc = history_feature_extract.history['val_accuracy']
    loss = history_feature_extract.history['loss']
    val_loss = history_feature_extract.history['val_loss']

# Accuracy Curve
plt.figure(figsize=(10, 5))
plt.plot(acc, label="Train Accuracy", color='blue')
plt.plot(val_acc, label="Validation Accuracy", color='orange')
plt.axvline(x=INITIAL_EPOCHS-1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("training_accuracy.png", dpi=300)
print("Saved: training_accuracy.png")

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(loss, label="Train Loss", color='blue')
plt.plot(val_loss, label="Validation Loss", color='orange')
plt.axvline(x=INITIAL_EPOCHS-1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png", dpi=300)
print("Saved: training_loss.png")

print(f"\n✅ Training Complete. Best model saved to: {MODEL_PATH}")
