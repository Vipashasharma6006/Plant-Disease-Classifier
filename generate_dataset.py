import os
import numpy as np
from PIL import Image, ImageDraw
import random

# Disease folders to create
diseases = ['Healthy', 'Bacterial_Spot', 'Late_Blight', 'Powdery_Mildew']
num_images = 100
img_size = (128, 128)

# Create main folder
if not os.path.exists("synthetic_leaf_dataset"):
    os.mkdir("synthetic_leaf_dataset")

# Function to draw leaf with spots
def draw_leaf(label):
    img = Image.new("RGB", img_size, (34, 139, 34))  # green background
    draw = ImageDraw.Draw(img)

    if label != "Healthy":
        for _ in range(random.randint(3, 8)):
            x, y = random.randint(10, 100), random.randint(10, 100)
            r = random.randint(5, 15)

            color = {
                "Bacterial_Spot": (139, 69, 19),     # brown
                "Late_Blight": (128, 0, 0),          # dark red
                "Powdery_Mildew": (255, 255, 255),   # white
            }.get(label, (0, 0, 0))

            draw.ellipse((x, y, x + r, y + r), fill=color)

    return img

# Generate folders and images
for disease in diseases:
    folder_path = f"synthetic_leaf_dataset/{disease}"
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_images):
        image = draw_leaf(disease)
        image.save(f"{folder_path}/{disease}_{i}.jpg")

print("✅ Synthetic leaf image dataset created!")
