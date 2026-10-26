
import os
import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------- SETTINGS ----------------

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

REAL_DIR = os.path.join("dataset", "processed", "real")
FAKE_DIR = os.path.join("dataset", "processed", "fake")

CNN_PATH = "forgery_detection_model_v3.keras"
HYBRID_MODEL_PATH = "hybrid_forgery_classifier.pkl"

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# ---------------- CHECK FILES ----------------

if not os.path.exists(CNN_PATH):
    raise FileNotFoundError(f"CNN model not found: {CNN_PATH}")

for folder in [REAL_DIR, FAKE_DIR]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

# ---------------- LOAD DATASET PATHS ----------------

fake_paths = sorted(
    os.path.join(FAKE_DIR, f)
    for f in os.listdir(FAKE_DIR)
    if f.lower().endswith(EXTENSIONS)
)

real_paths = sorted(
    os.path.join(REAL_DIR, f)
    for f in os.listdir(REAL_DIR)
    if f.lower().endswith(EXTENSIONS)
)

# Fake = 0, Real = 1
paths = np.array(fake_paths + real_paths)
labels = np.array(
    [0] * len(fake_paths) + [1] * len(real_paths),
    dtype=np.int32
)

print(f"Fake images: {len(fake_paths)}")
print(f"Real images: {len(real_paths)}")
print(f"Total images: {len(paths)}")

# ---------------- SPLIT DATA ----------------

train_paths, val_paths, y_train, y_val = train_test_split(
    paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

# ---------------- LOAD CNN ----------------

print("\nLoading trained CNN...")
cnn = tf.keras.models.load_model(CNN_PATH)

# Use the CNN's learned 64-dimensional representation.
feature_layer = None

for layer in cnn.layers:
    if isinstance(layer, tf.keras.layers.Dense) and layer.units == 64:
        feature_layer = layer
        break

if feature_layer is None:
    raise ValueError(
        "Could not find the CNN's 64-unit feature layer. "
        "Please share the model layer names."
    )

feature_extractor = tf.keras.Model(
    inputs=cnn.inputs,
    outputs=feature_layer.output
)

# ---------------- EXTRACT FEATURES ----------------

def extract_features(image_paths):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def load_image(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMG_SIZE)
        return tf.cast(image, tf.float32)

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    batches = []

    for images in dataset:
        features = feature_extractor(images, training=False)
        batches.append(features.numpy())

    return np.concatenate(batches, axis=0)

print("\nExtracting training features...")
X_train = extract_features(train_paths)
print("Training feature shape:", X_train.shape)

print("\nExtracting validation features...")
X_val = extract_features(val_paths)
print("Validation feature shape:", X_val.shape)

# ---------------- TRAIN RANDOM FOREST ----------------

print("\nTraining Random Forest classifier...")

classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

classifier.fit(X_train, y_train)

# ---------------- EVALUATE ----------------

print("\nEvaluating hybrid model...")
y_pred = classifier.predict(X_val)

print(f"\nHybrid Model Accuracy: {accuracy_score(y_val, y_pred):.2%}")

print("\nConfusion Matrix (Fake=0, Real=1):")
print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

# ---------------- SAVE HYBRID CLASSIFIER ----------------

joblib.dump(classifier, HYBRID_MODEL_PATH)

print(f"\nHybrid classifier saved: {HYBRID_MODEL_PATH}")
print("Hybrid training and evaluation complete.")
