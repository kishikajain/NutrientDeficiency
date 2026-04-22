# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json

# # Load model
# model = tf.keras.models.load_model("models/nutrient_model.h5")

# # Load class names
# with open("models/class_names.json", "r") as f:
#     class_indices = json.load(f)

# # Reverse mapping (index → class name)
# class_names = {v: k for k, v in class_indices.items()}

# # Load image
# img_path = "test.jpg"   # change to your image path
# image = Image.open(img_path)

# # Preprocess
# image = image.resize((224, 224))
# img_array = np.array(image) / 255.0
# img_array = np.expand_dims(img_array, axis=0)

# # Predict
# prediction = model.predict(img_array)
# predicted_index = np.argmax(prediction)
# predicted_class = class_names[predicted_index]
# confidence = np.max(prediction) * 100

# # Output
# print("Prediction:", predicted_class)
# print("Confidence:", f"{confidence:.2f}%")
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("models/nutrient_model.h5")

# Load class names
with open("models/class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

# ---------------- PREPROCESS ----------------
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------- PREDICT ----------------
def predict(image_path):
    img_array = preprocess(image_path)

    prediction = model.predict(img_array)

    top3_idx = np.argsort(prediction[0])[-3:][::-1]

    confidence = np.max(prediction) * 100
    predicted_class = class_names[np.argmax(prediction)]

    # confidence logic
    if confidence < 60:
        predicted_class = "Low confidence / Unknown"

    return predicted_class, confidence, top3_idx, prediction[0]


# Example run
if __name__ == "__main__":
    img_path = "test.jpg"

    label, conf, top3_idx, probs = predict(img_path)

    print("\nPrediction:", label)
    print("Confidence:", f"{conf:.2f}%")

    print("\nTop 3 Predictions:")
    for i in top3_idx:
        print(class_names[i], f"{probs[i]*100:.2f}%")