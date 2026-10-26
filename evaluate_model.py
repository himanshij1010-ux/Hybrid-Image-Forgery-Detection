
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

# -----------------------------
# 1. Configuration
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

MODEL_PATH = "forgery_detection_model_v3.keras"
REAL_DIR = os.path.join("dataset", "processed", "real")
FAKE_DIR = os.path.join("dataset", "processed", "fake")

# -----------------------------
# 2. Check required files
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.isdir(REAL_DIR) or not os.path.isdir(FAKE_DIR):
    raise FileNotFoundError(
        "Dataset folders not found. Check dataset/processed/real "
        "and dataset/processed/fake."
    )

# -----------------------------
# 3. Collect image paths
# -----------------------------
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

real_paths = sorted(
    os.path.join(REAL_DIR, name)
    for name in os.listdir(REAL_DIR)
    if name.lower().endswith(extensions)
)

fake_paths = sorted(
    os.path.join(FAKE_DIR, name)
    for name in os.listdir(FAKE_DIR)
    if name.lower().endswith(extensions)
)

# Labels used during training: Fake = 0, Real = 1
paths = np.array(fake_paths + real_paths)
labels = np.array(
    [0] * len(fake_paths) + [1] * len(real_paths),
    dtype=np.int32
)

print(f"Total fake images: {len(fake_paths)}")
print(f"Total real images: {len(real_paths)}")

# -----------------------------
# 4. Recreate validation split
# -----------------------------
_, val_paths, _, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print(f"Validation fake images: {(val_labels == 0).sum()}")
print(f"Validation real images: {(val_labels == 1).sum()}")

# -----------------------------
# 5. Prepare validation dataset
# -----------------------------
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 6. Load model and predict
# -----------------------------
print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Generating predictions...")
y_true = np.array(val_labels, dtype=int)

probabilities = model.predict(
    validation_dataset,
    verbose=1
).ravel()

# -----------------------------
# 7. Compare decision thresholds
# -----------------------------
print("\n========== THRESHOLD COMPARISON ==========")
print("A lower threshold generally predicts more images as Fake.")

for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    predictions = (probabilities >= threshold).astype(int)

    fake_recall = (
        ((predictions == 0) & (y_true == 0)).sum()
        / max((y_true == 0).sum(), 1)
    )

    fake_precision = (
        ((predictions == 0) & (y_true == 0)).sum()
        / max((predictions == 0).sum(), 1)
    )

    fake_f1 = f1_score(
        y_true,
        predictions,
        pos_label=0,
        zero_division=0
    )

    accuracy = accuracy_score(y_true, predictions)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {accuracy:.2%} | "
        f"Fake Precision: {fake_precision:.2%} | "
        f"Fake Recall: {fake_recall:.2%} | "
        f"Fake F1: {fake_f1:.3f}"
    )

# -----------------------------
# 8. Standard evaluation at 0.5
# -----------------------------
y_pred = (probabilities >= 0.5).astype(int)

print("\n========== MODEL EVALUATION AT 0.50 ==========")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

print("\nEvaluation complete.")
