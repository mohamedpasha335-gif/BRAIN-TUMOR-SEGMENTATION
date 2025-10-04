"""Streamlit app for running segmentation inference.
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

import losses  # so dice_loss etc are available if loading custom model
from unet import create_unet

st.title('Segmentation Inference - Brain Tumor (example)')

uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg','tif','tiff'])
model_path = st.text_input('Model path (h5)', 'unet_segmentation.h5')

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    size = (128,128)
    img_resized = img.resize(size)
    img_arr = np.array(img_resized).astype('float32')/255.0
    # if model expects single channel, convert to grayscale
    model = None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.warning(f"Could not load model directly: {e}\nTrying to create architecture and load weights...")
        try:
            # try to build the architecture and load weights
            m = create_unet(input_size=(128,128,3))
            m.load_weights(model_path)
            model = m
        except Exception as e2:
            st.error(f"Failed to load model or weights: {e2}")
    if model is not None:
        inp = np.expand_dims(img_arr, 0)
        pred = model.predict(inp)[0]
        # resize prediction to original size for display
        pred_resized = cv2.resize(pred.squeeze(), img.size[::-1])
        st.image(img, caption='Input image', use_column_width=True)
        st.image(pred_resized, caption='Predicted mask', clamp=True, use_column_width=True)
