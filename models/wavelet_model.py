import pywt
import numpy as np
import cv2

def extract_features(image):
    """
    Extract features using 2D Complex Wavelet Transform (2D CWT).

    Args:
        image (numpy.ndarray): Preprocessed iris image.

    Returns:
        numpy.ndarray: Feature vector.
    """
    # Make sure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image for consistency
    image = cv2.resize(image, (128, 128))

    # Apply 2D Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')  # You can choose other wavelets like 'db2', 'sym3' if you want
    cA, (cH, cV, cD) = coeffs

    # Flatten and concatenate features
    features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))

    return features
