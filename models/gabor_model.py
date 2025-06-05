# models/gabor_model.py
import numpy as np
import cv2
import math

def log_gabor_filter(shape, omega0=0.5, sigmaF=0.55):
    """Generate Log-Gabor filter in frequency domain."""
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)
    y = np.linspace(-0.5, 0.5, rows)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    radius[int(rows/2), int(cols/2)] = 1  # Avoid log(0)

    log_gabor = np.exp((-(np.log(radius / omega0))**2) / (2 * (np.log(sigmaF))**2))
    log_gabor[int(rows/2), int(cols/2)] = 0  # Center frequency set to zero
    return log_gabor

def extract_features(image):
    """Apply Log-Gabor filter and extract features."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    gabor = log_gabor_filter(image.shape)
    filtered = np.multiply(f_shift, gabor)

    filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered))
    magnitude = np.abs(filtered_img)

    # Flatten the features
    features = magnitude.flatten()

    return features
