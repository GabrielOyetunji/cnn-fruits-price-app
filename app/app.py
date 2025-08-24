# app/app.py
import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# -----------------------------------------------------------------------------
# Paths (repo layout: /app/app.py, /model/multitask_model.keras, /model/meta.json)
# -----------------------------------------------------------------------------
# _file_ (two underscores!) gives the current file's path.
REPO_ROOT = Path(_file_).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
MODEL_PATH = MODEL_DIR / "multitask_model.keras"
META_PATH = MODEL_DIR / "meta.json"

# -----------------------------------------------------------------------------
# Load model + metadata (cached so it loads only once per server session)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_meta():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta file not found at: {META_PATH}")

    # compile=False is fine for inference and loads faster
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    img_size = int(meta.get("img_size", 224))
    return model, class_names, img_size

model, class_names, IMG_SIZE = load_model_and_meta()

# -----------------------------------------------------------------------------
# Preprocessing to match training (resize to IMG_SIZE and scale to [0,1])
# -----------------------------------------------------------------------------
def preprocess_image(img: Image.Image, img_size: int):
    img = img.convert("RGB").resize((img_size, img_size))  # match training size
    x = np.array(img, dtype=np.float32) / 255.0            # Rescaling(1/255)
    x = np.expand_dims(x, axis=0)                          # (1, H, W, 3)
    return x, img

# -----------------------------------------------------------------------------
# Single-image prediction helper
# -----------------------------------------------------------------------------
def run_prediction(img: Image.Image):
    x, display_img = preprocess_image(img, IMG_SIZE)

    # Your saved model has two outputs: [class_probs, price_pred]
    class_probs, price_pred = model.predict(x, verbose=0)

    # Classification post-processing
    probs = class_probs[0]
    class_idx = int(np.argmax(probs))
    class_name = class_names[class_idx]
    top5_idx = np.argsort(-probs)[:5]
    top5 = [(class_names[i], float(probs[i])) for i in top5_idx]

    # Price post-processing
    price_value = float(price_pred[0][0])

    return class_name, price_value, top5, display_img

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fruit Classifier & Price Predictor", page_icon="🍊", layout="centered")

st.title("🍉 Fruit Classification & Price Prediction")
st.write("Upload a fruit/vegetable image — the model will recognize the item and predict its retail price.")

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png, webp)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Running inference…"):
            class_name, price_value, top5, display_img = run_prediction(image)

        st.subheader("Prediction")
        st.markdown(f"*Class:* {class_name}")
        st.markdown(f"*Estimated Price:* ${price_value:.2f}")

        st.subheader("Top-5 class probabilities")
        for nm, prob in top5:
            st.write(f"- {nm}: {prob:.3f}")
