import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model once
@st.cache_resource
def load_my_model():
    return load_model('outputs/models/new_best_model.h5', compile=True)

model = load_my_model()

# Preprocessing
def preprocess_image_pil(img: Image.Image, target_size=(128, 128)):
    img = img.convert('L').resize(target_size)  # Grayscale
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 1)
    img_array = img_array / 255.0
    return img_array

# Interface
st.title("🎂 Age Prediction")
st.write("Upload an image or use your camera to predict age.")

# Tabs for upload or camera
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Predicting..."):
            img_array = preprocess_image_pil(img)
            pred = model.predict(img_array)
            # If the model output is a single age value (shape: (1, 1))
            pred_age = round(pred[0][0])
            st.success(f"🎂 Predicted Age: {pred_age}")

with tab2:
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image", use_column_width=True)

        with st.spinner("Predicting..."):
            img_array = preprocess_image_pil(img)
            pred = model.predict(img_array)
            pred_age = round(pred[0][0])
            st.success(f"🎂 Predicted Age: {pred_age}")
