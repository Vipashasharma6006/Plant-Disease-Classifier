import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("leaf_cnn_model.h5")
#class_names = ['Bacterial_Spot', 'Healthy', 'Late_Blight', 'Powdery_Mildew']
import os
class_names = sorted(os.listdir('synthetic_leaf_dataset'))


st.title("🌿 Plant Disease Classifier")
st.write("Upload a leaf image to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"🌱 Predicted Disease: **{predicted_class}**")
    st.info(f"🧠 Confidence: `{confidence:.2f}`")
