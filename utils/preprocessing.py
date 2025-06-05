# utils/preprocessing.py
import cv2
import numpy as np

def preprocess_image(img):
    if img is None:
        raise ValueError("Input image is None")
    if not isinstance(img, np.ndarray):
        raise TypeError("Expected a NumPy array")
    if img.ndim not in [2, 3]:
        raise ValueError(f"Unexpected image dimensions: {img.ndim}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    return img
