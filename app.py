
import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

from PIL import Image
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Hybrid Image Forgery Detection",
    page_icon="🔍",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
CNN_PATH = BASE_DIR / "forgery_detection_model_v3.keras"
CLASSIFIER_PATH = BASE_DIR / "hybrid_forgery_classifier.pkl"

IMG_SIZE = (128, 128)

st.title("🔍 Hybrid Image Forgery Detection")
st.write(
    "This application combines CNN-based feature extraction "
    "with a Random Forest classifier to analyze images."
)

st.info(
    "Experimental project: predictions can be incorrect and "
    "should not be treated as proof of authenticity."
)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model(CNN_PATH)

    feature_layer = None
    for layer in cnn.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 64:
            feature_layer = layer
            break

    if feature_layer is None:
        raise ValueError("CNN feature layer could not be found.")

    feature_extractor = tf.keras.Model(
        inputs=cnn.inputs,
        outputs=feature_layer.output
    )

    classifier = joblib.load(CLASSIFIER_PATH)

    return feature_extractor, classifier


if not CNN_PATH.exists():
    st.error("CNN model file is missing.")
    st.stop()

if not CLASSIFIER_PATH.exists():
    st.error("Hybrid classifier file is missing.")
    st.stop()

try:
    feature_extractor, classifier = load_models()
except Exception as error:
    st.error(f"Could not load the models: {error}")
    st.stop()

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload an image for analysis",
    type=["jpg", "jpeg", "png", "bmp", "gif"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch"
        )

        resized_image = image.resize(IMG_SIZE)

        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(image_array, axis=0)

        with st.spinner("Analyzing image with the hybrid model..."):
            features = feature_extractor(
                image_array,
                training=False
            ).numpy()

            prediction = int(classifier.predict(features)[0])
            probabilities = classifier.predict_proba(features)[0]

        # Training labels: Fake = 0, Real = 1
        class_indices = list(classifier.classes_)
        predicted_index = class_indices.index(prediction)
        confidence = float(probabilities[predicted_index]) * 100

        if prediction == 1:
            label = "Real"
            st.success("Predicted class: Real")
        else:
            label = "Fake"
            st.warning("Predicted class: Fake")

        st.metric(
            "Predicted class",
            label
        )

        st.metric(
            "Classifier score for predicted class",
            f"{confidence:.2f}%"
        )

        st.caption(
            "The displayed score is the Random Forest classifier's "
            "estimated class probability. It is not a calibrated "
            "probability that the image is genuinely real or fake."
        )

        st.subheader("Class Scores")

        for class_value, probability in zip(
            classifier.classes_,
            probabilities
        ):
            class_name = "Fake" if class_value == 0 else "Real"
            st.write(f"{class_name}: {probability * 100:.2f}%")

    except Exception as error:
        st.error(f"Could not analyze the image: {error}")

else:
    st.write("Upload an image above to begin.")
