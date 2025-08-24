# app/app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image

# --- Paths ---
MODEL_DIR = Path(_file_).resolve().parents[1] / "model"
MODEL_PATH = MODEL_DIR / "multitask_model.keras"
META_PATH = MODEL_DIR / "meta.json"

# --- Load model and metadata ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    img_size = int(meta.get("img_size", 224))
    return model, class_names, img_size

model, class_names, IMG_SIZE = load_model()

# --- Preprocess helper ---
def preprocess_image(img: Image.Image, img_size: int):
    img = img.convert("RGB").resize((img_size, img_size))
    x = np.array(img) / 255.0
    return np.expand_dims(x, axis=0), img

# --- Prediction helper ---
def predict(img: Image.Image):
    x, pil_img = preprocess_image(img, IMG_SIZE)
    class_probs, price_pred = model.predict(x, verbose=0)

    # Classification
    class_idx = int(np.argmax(class_probs[0]))
    class_name = class_names[class_idx]
    top5_idx = np.argsort(-class_probs[0])[:5]
    top5 = [(class_names[i], float(class_probs[0][i])) for i in top5_idx]

    # Price prediction
    price_value = float(price_pred[0][0])
    return class_name, price_value, top5, pil_img

# --- Streamlit UI ---
st.set_page_config(page_title="Fruit Classifier & Price Predictor", page_icon="🍊", layout="centered")
st.title("🍉 Fruit Classification & Price Prediction")
st.write("Upload an image of a fruit/vegetable and the model will recognize it and predict its retail price.")

uploaded_file = st.file_uploader("Upload an image (jpg, png, jpeg, webp)", type=["jpg","png","jpeg","webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        class_name, price_value, top5, pil_img = predict(image)

        st.subheader("Prediction Results")
        st.write(f"*Predicted Class:* {class_name}")
        st.write(f"*Predicted Price:* ${price_value:.2f}")

        st.markdown("*Top-5 Class Probabilities:*")
        for nm, prob in top5:
            st.write(f"- {nm}: {prob:.3f}")