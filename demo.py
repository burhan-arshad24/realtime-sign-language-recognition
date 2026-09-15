import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("sign_language_model.keras")
class_names = joblib.load("sign_class_names.pkl")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ROI_TOP = 80
ROI_BOTTOM = 400
ROI_LEFT = 160
ROI_RIGHT = 480

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    roi = frame[
        ROI_TOP:ROI_BOTTOM,
        ROI_LEFT:ROI_RIGHT
    ]

    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb_roi, (64, 64))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    prediction = model(img, training=False).numpy()[0]

    top_indices = np.argsort(prediction)[-5:][::-1]

    predicted_class = class_names[top_indices[0]]
    confidence = float(prediction[top_indices[0]])

    cv2.rectangle(
        frame,
        (ROI_LEFT, ROI_TOP),
        (ROI_RIGHT, ROI_BOTTOM),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{predicted_class} {confidence:.1%}",
        (ROI_LEFT, ROI_TOP - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    y = 30

    for i in top_indices:
        text = f"{class_names[i]}: {prediction[i]:.2%}"

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y += 25

    cv2.imshow("ASL Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("webcam_input.jpg", roi)

        print("\n========== SAVED IMAGE ==========")
        print("File: webcam_input.jpg")
        print("ROI shape:", roi.shape)
        print("Model input shape:", img.shape)
        print("Pixel min:", img.min())
        print("Pixel max:", img.max())
        print("Pixel mean:", img.mean())

        print("\nTop 5 predictions:")
        for i in top_indices:
            print(
                f"{class_names[i]}: "
                f"{prediction[i]:.4f}"
            )

        print("================================\n")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()