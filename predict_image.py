import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model('leaf_cnn_model.h5')

# Class labels (ensure this matches training order)
class_names = ['Bacterial_Spot', 'Healthy', 'Late_Blight', 'Powdery_Mildew']

# Load a sample image (you can change the path here!)
img_path = 'synthetic_leaf_dataset/Late_Blight/Late_Blight_5.jpg'

# Load & preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

# Result
print(f"\n🌿 Predicted Disease: {predicted_class}")
