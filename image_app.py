from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import cv2

app = FastAPI(title="MNIST Digit Prediction API")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_digit_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "MNIST Digit Prediction API is running"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Convert PIL to NumPy
    image_np = np.array(image)

    # Preprocessing for phone camera images
    image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
    _, image_np = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)

    # Resize to MNIST size
    image_np = cv2.resize(image_np, (28, 28))

    # Normalize
    image_np = image_np / 255.0

    # Reshape for model
    image_np = image_np.reshape(1, 28, 28)

    # Predict
    predictions = model.predict(image_np)
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4)
    }

