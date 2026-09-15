# ASL Alphabet Sign Language Recognition

An end-to-end deep learning application that recognizes American Sign Language (ASL) alphabet gestures from live camera frames using a custom Convolutional Neural Network (CNN).

The project includes a trained TensorFlow/Keras model, a FastAPI backend, WebSocket-based real-time prediction, and a browser-based live camera interface.

## Live Demo

**Live Demo:** https://realtime-sign-language-recognition-burhan.onrender.com/


## Features

* Real-time ASL alphabet recognition
* 29-class classification
* A-Z alphabet recognition
* `del`, `nothing`, and `space` classes
* Custom CNN built with TensorFlow/Keras
* Real-time predictions through FastAPI WebSockets
* Browser-based camera interface
* Confidence score for predictions
* FastAPI serves both frontend and backend
* Production-ready project structure

## Problem

American Sign Language uses visual hand gestures to communicate. This project focuses on recognizing individual ASL alphabet gestures from camera images.

The model predicts one of 29 classes:

* A-Z
* del
* nothing
* space

The goal was to build a complete computer vision pipeline from dataset preparation and model training to real-time inference through a web application.

## Dataset

The model was trained using the Kaggle ASL Alphabet dataset by `grassknoted`.

Dataset characteristics:

* 87,000 training images
* 29 classes
* Approximately 3,000 images per class
* Original image resolution: 200 x 200
* Images resized to 64 x 64 for training

The dataset is not included in this repository.

## Model

A custom Convolutional Neural Network was developed using TensorFlow/Keras.

### Architecture

```text
Input: 64 x 64 x 3
        ↓
Random Rotation
        ↓
Random Zoom
        ↓
Random Brightness
        ↓
Random Contrast
        ↓
Rescaling
        ↓
Conv2D: 32 filters
        ↓
MaxPooling
        ↓
Conv2D: 64 filters
        ↓
MaxPooling
        ↓
Conv2D: 64 filters
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense: 128
        ↓
Dropout: 0.4
        ↓
Dense: 29
        ↓
Softmax
```

The model contains its own `Rescaling(1./255)` layer, so the API does not apply a second normalization step during inference.

## Performance

The model achieved approximately **99.19% validation accuracy** on a 20% validation split containing 17,400 images.

### Validation Results

* Validation Accuracy: **99.19%**
* Validation Loss: **0.0250**

The majority of classes achieved near-perfect precision, recall, and F1 scores.

The relatively weaker classes were R, U, S, and V, but their performance remained strong.

## Real-Time Recognition Pipeline

```text
Browser Camera
      ↓
320 x 320 ROI
      ↓
JPEG Encoding
      ↓
Base64
      ↓
WebSocket
      ↓
FastAPI
      ↓
OpenCV Image Decode
      ↓
BGR → RGB
      ↓
Resize to 64 x 64
      ↓
TensorFlow/Keras CNN
      ↓
29-Class Prediction
      ↓
Confidence Score
      ↓
WebSocket Response
      ↓
Browser UI
```

The browser captures a region of the camera feed and sends it to the FastAPI WebSocket endpoint.

The backend decodes the image, converts it from BGR to RGB, resizes it to the model's required input size, and performs inference.

The predicted class and confidence score are then returned to the browser in real time.

## Technologies

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Joblib
* FastAPI
* WebSockets
* HTML
* CSS
* JavaScript

## Project Structure

```text
Sign Language Recognition/
│
├── main.py
├── sign_language_model.keras
├── sign_class_names.pkl
├── requirements.txt
├── runtime.txt
├── .gitignore
│
└── static/
    └── index.html
```

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/burhan-arshad24/realtime-sign-language-recognition
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start FastAPI

```bash
uvicorn main:app --reload
```

### 5. Open the Application

Open:

```text
http://127.0.0.1:8000/
```

Allow browser camera access when prompted.

## API

### Frontend

```text
GET /
```

Serves the web application from `static/index.html`.

### WebSocket

```text
/ws/predict
```

The frontend sends Base64-encoded JPEG frames through the WebSocket.

The backend returns:

```json
{
    "letter": "A",
    "confidence": 0.998
}
```

## Deployment

The application can be deployed to a cloud platform that supports Python web services and WebSockets.

The production server should run:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The deployment platform must support WebSocket connections because live predictions are sent through a WebSocket.

HTTPS is also required for reliable browser camera access in a deployed environment.

## Deployment Requirements

The following files are required at runtime:

```text
main.py
sign_language_model.keras
sign_class_names.pkl
requirements.txt
runtime.txt
static/index.html
```

The original training dataset is not required for inference.

## Important Notes

The application uses the trained Keras model directly for inference.

The model already performs pixel rescaling internally through its `Rescaling(1./255)` layer.

The backend also converts OpenCV's BGR image format to RGB before inference so that the input format matches the training pipeline.

The frontend automatically uses:

```text
ws://
```

for local HTTP development and:

```text
wss://
```

when deployed over HTTPS.

## Limitations

* The current system recognizes individual ASL alphabet gestures rather than complete words or sentences.
* Performance can vary depending on lighting conditions.
* Camera quality can affect predictions.
* Background and hand position can influence recognition.
* The model was trained on the ASL Alphabet dataset and may not generalize perfectly to every real-world environment.
* Real-time performance depends on the deployment server and network connection.

## Future Improvements

* Automatic hand detection
* MediaPipe hand tracking
* Automatic ROI extraction
* Word-level sign recognition
* Sentence-level sign recognition
* Temporal gesture recognition using LSTM/GRU/Transformer models
* Text-to-speech output
* Mobile application
* Multi-language sign recognition
* Improved robustness to lighting and backgrounds

## Author

**Burhan Arshad**

AI / Machine Learning Developer
