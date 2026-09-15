from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import cv2
import base64
import joblib
from tensorflow.keras.models import load_model


app = FastAPI(title="Sign Language Recognition")

@app.get("/health")
def health():
    return {"status": "ok"}

model = load_model("sign_language_model.keras")
class_names = joblib.load("sign_class_names.pkl")


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)



@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


def predict_frame(frame_bytes):
    nparr = np.frombuffer(frame_bytes, np.uint8)

    frame = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise ValueError("Could not decode image")

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        frame,
        (64, 64)
    )

    img = img.astype(np.float32)

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model(
        img,
        training=False
    ).numpy()[0]

    index = int(np.argmax(prediction))

    predicted_class = class_names[index]
    confidence = float(prediction[index])

    return predicted_class, confidence


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):

    await websocket.accept()

    try:

        while True:

            data = await websocket.receive_text()

            if "," in data:
                data = data.split(",", 1)[1]

            img_bytes = base64.b64decode(data)

            predicted_class, confidence = predict_frame(
                img_bytes
            )

            await websocket.send_json(
                {
                    "letter": predicted_class,
                    "confidence": round(confidence, 3)
                }
            )

    except WebSocketDisconnect:

        print("Client disconnected")

    except Exception as e:

        print(f"Error: {e}")