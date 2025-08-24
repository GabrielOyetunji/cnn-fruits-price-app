# app/app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image

# --- Resolve repo root and model paths (uses __file__, not _file_) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR  = REPO_ROOT / "model"
MODEL_PATH = MODEL_DIR / "multitask_model.keras"
META_PATH  = MODEL_DIR / "meta.json"

# --- UI config ---
st.set_page_config(page_title="Fruit Classifier & Price Predictor", page_icon="🍊", layout="centered")
st.title("🍉 Fruit Classification & Price Prediction")
st.write("Upload an image of a fruit/vegetable. The model will recognize the class and predict its retail price.")

# --- Safe checks so the app fails gracefully if files are missing ---
if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        f"Model files not found.\n\n"
        f"Expected:\n- {MODEL_PATH}\n- {META_PATH}\n\n"
        f"Make sure the repo has a `model/` folder containing `multitask_model.keras` and `meta.json`."
    )
    st.stop()

# --- Load model and metadata (cached) ---
@st.cache_resource
def load_model_and_meta():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    class_names = meta["class_names"]
    img_size = int(meta.get("img_size", 224))
    return model, class_names, img_size

model, class_names, IMG_SIZE = load_model_and_meta()

# --- Preprocess helper (match training Rescaling(1/255)) ---
def preprocess_image(img: Image.Image, img_size: int):
    img = img.convert("RGB").resize((img_size, img_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0), img  # (1, H, W, 3), PIL image

# --- Prediction helper ---
def run_inference(img: Image.Image):
    x, pil_img = preprocess_image(img, IMG_SIZE)
    class_probs, price_pred = model.predict(x, verbose=0)

    # Classification
    class_idx = int(np.argmax(class_probs[0]))
    class_name = class_names[class_idx]
    top5_idx = np.argsort(-class_probs[0])[:5]
    top5 = [(class_names[i], float(class_probs[0][i])) for i in top5_idx]

    # Price regression
    price_value = float(price_pred[0][0])
    return class_name, price_value, top5, pil_img

# --- UI: uploader + predict button ---
uploaded = st.file_uploader("Upload an image (jpg, png, jpeg, webp)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Running inference..."):
            class_name, price_value, top5, _ = run_inference(image)

        st.subheader("Prediction")
        st.write(f"**Class:** {class_name}")
        st.write(f"**Predicted Price:** ${price_value:.2f}")

        st.markdown("**Top-5 Class Probabilities**")
        for nm, prob in top5:
            st.write(f"- {nm}: {prob:.3f}")
else:
    st.info("👆 Upload an image to begin.")
