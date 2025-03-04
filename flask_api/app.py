import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Layer

# Initialize Flask app
app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

# Load trained model
MODEL_PATH = os.path.abspath("models/disease_model_balanced.h5")  # Ensure correct path
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model file not found:", MODEL_PATH)

# Define custom Cast layer to avoid deserialization error
class Cast(Layer):
    def call(self, inputs):
        return tf.cast(inputs, dtype=tf.float32)

# Handle missing activation and custom Cast layer issue
custom_objects = {"Activation": Activation, "Cast": Cast}

try:
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Loading Error:", e)
    model = None  # Avoid crashes if model loading fails

# Class labels (Ensure this matches model's output layer size)
class_labels = [
    "Stroke Risk", "High Blood Pressure", "Cardiovascular Issues", 
    "Migraine", "Mental Health Disorder", "Cognitive Decline", "Dementia"
]  # ⚠️ Ensure length matches model output nodes

# Route: Home Page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Route: Predict Disease from Palm Image
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    if img_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image as BytesIO
        img_bytes = io.BytesIO(img_file.read())

        # Load and preprocess the image
        img = image.load_img(img_bytes, target_size=(160, 160))  # Match model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand for batch

        # Ensure model is loaded before predicting
        if model is None:
            return jsonify({"error": "Model not loaded. Check logs."}), 500

        # Predict disease
        predictions = model.predict(img_array)
        print("Raw Predictions:", predictions)  # Debugging log

        predicted_class = np.argmax(predictions, axis=1)[0]

        # Ensure predicted_class is within range of class_labels
        if predicted_class >= len(class_labels):
            print("⚠️ Predicted class index out of range:", predicted_class)
            return jsonify({"error": "Predicted class index out of range"}), 500

        disease_name = class_labels[predicted_class]
        confidence = float(predictions[0][predicted_class])  # Convert confidence score

        return jsonify({"predicted_disease": disease_name, "confidence": confidence})

    except Exception as e:
        import traceback
        print("Prediction Error:", traceback.format_exc())  # Log full error
        return jsonify({"error": "Prediction failed. Check logs for details."}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)