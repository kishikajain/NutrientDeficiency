import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vitamin Deficiency Detection",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/nutrient_model.h5")

model = load_model()

# ---------------- CLASS MAP ----------------
with open("models/class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------- TITLE ----------------
st.title("🥗 Vitamin Deficiency Detection System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---------------- MAIN UI ----------------
if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    # ---------------- LEFT SIDE ----------------
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    # ---------------- RIGHT SIDE ----------------
    with col2:
        st.subheader("Prediction Results")

        img_array = preprocess(image)
        prediction = model.predict(img_array)

        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        top3_idx = np.argsort(prediction[0])[-3:][::-1]

        predicted_class = class_names[predicted_index]

        # ---------------- CONFIDENCE ----------------
        if confidence < 60:
            st.warning("⚠ Low confidence prediction")
            predicted_class = "Unknown / Low confidence"

        # ---------------- OUTPUT ----------------
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.divider()

        st.subheader("Top 3 Predictions")

        for i in top3_idx:
            st.write(f"{class_names[i]} → {prediction[0][i]*100:.2f}%")