import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('3d_unet_model.h5')

st.title("3D U-Net Tumor Detection App")
uploaded_file = st.file_uploader("Upload a medical image slice", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128)).reshape(1, 128, 128, 1) / 255.0

    prediction = model.predict(image)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Tumor Probability: {prediction[0][0][0][0]:.2f}")
