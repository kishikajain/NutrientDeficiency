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

# Load image
img_path = "test.jpg"   # change to your image path
image = Image.open(img_path)

# Preprocess
image = image.resize((224, 224))
img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]
confidence = np.max(prediction) * 100

# Output
print("Prediction:", predicted_class)
print("Confidence:", f"{confidence:.2f}%")