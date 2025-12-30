🧠 MNIST Digit Classification API (TensorFlow + FastAPI)
📌 Project Overview

This project implements an end-to-end handwritten digit recognition system using TensorFlow for model training and FastAPI for deployment.
The trained model predicts digits (0–9) from the MNIST dataset and is exposed as a REST API capable of handling both JSON-based pixel inputs and real image uploads (including phone camera images).

The project demonstrates the complete machine learning lifecycle — from training and preprocessing to deployment and inference.

🚀 Key Features

✅ Trained a deep learning model on the MNIST dataset using TensorFlow/Keras

✅ REST API built with FastAPI

✅ Supports image upload via Swagger UI

✅ Handles phone camera images using OpenCV preprocessing

✅ Returns predicted digit with confidence score

✅ Production-style API with input validation and error handling

🛠️ Tech Stack
Category	Tools
Programming Language	Python
Deep Learning	TensorFlow / Keras
Image Processing	OpenCV, Pillow
Backend API	FastAPI
Server	Uvicorn
Numerical Computing	NumPy
📂 Project Structure
mnist_api/
│── image_app.py        # FastAPI application
│── mnist_model.h5      # Trained TensorFlow model
│── README.md           # Project documentation

🧠 Model Details

Dataset: MNIST (handwritten digits)

Input Shape: 28 × 28 grayscale images

Architecture:

Flatten Layer

Dense (ReLU)

Dense (Softmax)

Loss Function: Sparse Categorical Crossentropy

Optimizer: Adam

Output: Digit (0–9) with confidence score

🖼️ Image Preprocessing Pipeline

To support real-world images (including phone camera photos), the following preprocessing steps are applied:

Convert image to grayscale

Apply Gaussian blur to reduce noise

Thresholding and inversion

Resize to 28×28 (MNIST format)

Normalize pixel values (0–1)

Reshape for model input

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/mnist-digit-api.git
cd mnist-digit-api

2️⃣ Create & Activate Virtual Environment
python -m venv tf_env
tf_env\Scripts\activate    # Windows

3️⃣ Install Dependencies
pip install fastapi uvicorn tensorflow numpy pillow opencv-python

4️⃣ Run the API Server
uvicorn image_app:app --reload

5️⃣ Open Swagger UI
http://127.0.0.1:8000/docs

📤 API Endpoints
🔹 GET /

Health check endpoint.

Response

{
  "message": "MNIST Digit Prediction API is running"
}

🔹 POST /predict-image

Upload an image file (PNG/JPG) containing a handwritten digit.

Request

Content-Type: multipart/form-data

Upload a digit image

Response

{
  "predicted_digit": 5,
  "confidence": 0.93
}

🧪 Sample Use Cases

Handwritten digit recognition

OCR preprocessing pipelines

Image-based automation

Educational deep learning demos

Foundation for meter reading OCR systems

📈 Future Enhancements

🔹 Upgrade model to CNN for higher accuracy

🔹 Digit segmentation for multi-digit images

🔹 Smart meter reading OCR

🔹 Dockerize the API

🔹 Frontend interface for image upload

🔹 Model versioning & monitoring

🎯 Why This Project Matters

This project goes beyond a basic ML notebook by demonstrating:

✔ End-to-end ML workflow

✔ Model deployment

✔ Real-world image handling

✔ API-based inference

✔ Production-ready architecture

It is suitable for showcasing skills for:

Machine Learning Engineer

Data Scientist

AI Engineer

Computer Vision Engineer

👤 Author

Hrishikesh Shukla
Aspiring Data Scientist / ML Engineer

⭐ Acknowledgements

MNIST Dataset

TensorFlow & FastAPI documentation
