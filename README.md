# CNN Fruits Classification & Price Prediction App 🍊🍌🥦

This is a **Streamlit web app** that loads a pre-trained CNN model to:
- Classify fruit/vegetable images into categories
- Predict the retail price of the item
- Show top-5 most likely classes with probabilities

The app works by letting users **upload an image**, then the model returns:
- Predicted class name (e.g., Orange, Lettuce, Potato, etc.)
- Predicted price in dollars
- Top-5 predictions with confidence scores
- Visualization of the uploaded image with the prediction

---

## 📂 Project Structure
```
cnn-fruits-price-app/
│── app/                 # Streamlit app code (app.py)
│── model/               # Pre-trained model + meta
│   ├── multitask_model.keras
│   └── meta.json
│── README.md
│── requirements.txt
```

---

## ▶ Running the App

1. Clone this repository or download the folder:

```bash
git clone https://github.com/yourusername/cnn-fruits-price-app.git
cd cnn-fruits-price-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app/app.py
```

4. Upload an image (jpg/png) and get predictions 🎉

---

## 🔧 Requirements
- Python 3.9+
- TensorFlow (for loading the model)
- Streamlit (for web interface)
- NumPy, Pandas, Matplotlib

---

## ✨ Example Output
- **Predicted Class:** Orange  
- **Predicted Price:** $0.23  
- **Top-5 Predictions:** Orange (0.87), Grapes (0.05), Tomato (0.03), Banana (0.03), Lettuce (0.02)  
- **Visualization:** Image displayed with label and price  

---

## 📌 Notes
- Model was trained on a curated Fruits dataset with price mappings.
- Missing prices were masked during training, so predictions are available only for supported items.
- The model (`multitask_model.keras`) and metadata (`meta.json`) are already included in the `model/` folder.
