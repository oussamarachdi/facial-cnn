import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Load model once
@st.cache_resource
def load_my_model():
    return load_model('outputs/models/best_model.h5', compile=True)

model = load_my_model()

# Mapping for gender
gender_dict = {0: "Male", 1: "Female"}

# Preprocessing
def preprocess_image_pil(img: Image.Image, target_size=(128, 128)):
    img = img.convert('L').resize(target_size)  # Grayscale
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 1)
    img_array = img_array / 255.0
    return img_array

# Interface
st.title("👤 Age & Gender Prediction")
st.write("Upload an image or use your camera to predict age and gender.")

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
            pred_gender = gender_dict[round(pred[0][0][0])]
            pred_age = round(pred[1][0][0])
            st.success(f"👤 Gender: {pred_gender} | 🎂 Age: {pred_age}")

with tab2:
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image", use_column_width=True)

        with st.spinner("Predicting..."):
            img_array = preprocess_image_pil(img)
            pred = model.predict(img_array)
            pred_gender = gender_dict[round(pred[0][0][0])]
            pred_age = round(pred[1][0][0])
            st.success(f"👤 Gender: {pred_gender} | 🎂 Age: {pred_age}")
