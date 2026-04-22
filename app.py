import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("models/nutrient_model.h5")

# Load class names
with open("models/class_names.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index → class name)
class_names = {v: k for k, v in class_indices.items()}

# UI Title
st.title("🥗 Vitamin Deficiency Detection System")

st.write("Upload an image and the model will predict the deficiency.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")