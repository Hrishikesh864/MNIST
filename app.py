from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI(title="MNIST Digit Classifier API")

# Load trained model
model = tf.keras.models.load_model("mnist_digit_model.h5")

# Input schema
class PixelInput(BaseModel):
    pixel_values: list[float]

# Home route
@app.get("/")
def home():
    return {"message": "MNIST Digit Classifier API is running"}

# Prediction route
@app.post("/predict")
def predict_digit(data: PixelInput):
    # Convert list to numpy array
    pixels = np.array(data.pixel_values)

    # Validate input length
    if len(pixels) != 784:
        return {"error": "Input must contain exactly 784 pixel values"}

    # Reshape for model
    pixels = pixels.reshape(1, 28, 28)

    # Make prediction
    prediction = model.predict(pixels)

    # Extract result
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4)
    }
