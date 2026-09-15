
import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="centered"
)

model = load_model("sign_language_model.keras")
class_names = joblib.load("sign_class_names.pkl")

st.title("ASL Alphabet Recognition")
st.write("Upload an ASL hand gesture image to test the trained CNN model.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    img = image.resize((64, 64))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    prediction = model(img, training=False).numpy()[0]

    top_indices = np.argsort(prediction)[-5:][::-1]

    predicted_index = top_indices[0]
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Sign", predicted_class)

    with col2:
        st.metric("Confidence", f"{confidence:.2%}")

    st.subheader("Top 5 Predictions")

    for index in top_indices:
        st.write(
            f"**{class_names[index]}** — {prediction[index]:.2%}"
        )
