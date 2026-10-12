"""
Brain Tumor MRI Classification - Streamlit App
Run with:  streamlit run app.py
"""
import os
import json
import glob

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.60            # below this value the app asks for expert review

# Folders where the app looks for the saved .h5 models
MODEL_FOLDERS = ["models", "."]

# Default class order (same as the training notebook). It is overwritten by class_names.json if present.
DEFAULT_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": "A tumor that starts in the glial (supporting) cells of the brain or spinal cord. "
              "Gliomas vary from slow-growing to aggressive.",
    "Meningioma": "A tumor that grows from the meninges, the membranes covering the brain and spinal cord. "
                  "Most meningiomas are slow-growing.",
    "Pituitary": "A tumor in the pituitary gland at the base of the brain. It can affect hormone levels "
                 "and, if large, vision.",
    "No Tumor": "No tumor pattern was detected in this scan by the model.",
}


# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------
def find_model_files():
    # Return {display name: file path} for every .h5 file found
    files = {}
    for folder in MODEL_FOLDERS:
        for path in glob.glob(os.path.join(folder, "*.h5")):
            files[os.path.basename(path)] = path
    # Put best_model.h5 first so it is the default choice
    return dict(sorted(files.items(), key=lambda kv: (kv[0] != "best_model.h5", kv[0])))


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    # compile=False -> we only need predictions
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data
def load_class_names():
    for folder in MODEL_FOLDERS:
        path = os.path.join(folder, "class_names.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return DEFAULT_CLASSES


def preprocess(image: Image.Image) -> np.ndarray:
    # Same steps as in training: RGB -> resize 224x224 -> scale to 0-1
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
class_names = load_class_names()
model_files = find_model_files()

st.sidebar.title("⚙️ Settings")

if not model_files:
    st.sidebar.error("No .h5 model found. Put best_model.h5 in a 'models' folder next to app.py.")
    st.title("🧠 Brain Tumor MRI Classification")
    st.error("Model file not found. Copy `best_model.h5` and `class_names.json` into a `models/` folder and restart the app.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Choose a model", list(model_files.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write(
    "This app classifies a brain MRI image into one of four categories using a deep-learning model "
    "trained on the Brain Tumor MRI Multi-Class dataset."
)
st.sidebar.write("**Classes:** " + ", ".join(class_names))
st.sidebar.warning("For educational and decision-support use only. It is **not** a medical diagnosis.")

# --------------------------------------------------------------
# Main page
# --------------------------------------------------------------
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload a brain MRI image to predict the tumor type and see the model's confidence.")

model = load_model(model_files[selected_model_name])

uploaded_file = st.file_uploader("Upload an MRI image (JPG / JPEG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("👆 Please upload an MRI image to start.")
    st.stop()

image = Image.open(uploaded_file)

with st.spinner("Analyzing the image..."):
    probs = model.predict(preprocess(image), verbose=0)[0]

pred_idx = int(np.argmax(probs))
pred_class = class_names[pred_idx]
confidence = float(probs[pred_idx])

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Uploaded MRI")
    st.image(image)

with col2:
    st.subheader("Prediction")

    if pred_class == "No Tumor":
        st.success(f"✅ Predicted class: **{pred_class}**")
    else:
        st.error(f"⚠️ Predicted class: **{pred_class}**")

    st.metric("Confidence", f"{confidence * 100:.2f}%")
    st.progress(confidence)

    if confidence < LOW_CONFIDENCE:
        st.warning("Low confidence. Please have this scan reviewed by a radiologist.")

    st.markdown(f"**About {pred_class}:** {CLASS_INFO.get(pred_class, '')}")

    st.subheader("Confidence for every class")
    prob_df = pd.DataFrame({"Class": class_names, "Probability (%)": probs * 100}).set_index("Class")
    st.bar_chart(prob_df)

    with st.expander("Show probability table"):
        st.dataframe(prob_df.round(2))

st.markdown("---")
st.caption(f"Model used: {selected_model_name}  |  Input size: {IMG_SIZE[0]}x{IMG_SIZE[1]}  |  "
           "This tool supports, but never replaces, a qualified medical professional.")
