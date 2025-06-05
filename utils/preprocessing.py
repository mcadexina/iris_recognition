# utils/preprocessing.py
import cv2
import numpy as np

def preprocess_image(image_bytes):
    # Convert image bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 🧪 Apply basic preprocessing: normalization, noise reduction, etc
    img = cv2.resize(img, (224, 224))  # Example resizing
    img = img / 255.0  # Normalize

    return img
