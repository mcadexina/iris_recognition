import tensorflow as tf
from keras import layers, models
import numpy as np

# Create a CNN model for grayscale iris image classification or feature extraction
def create_cnn_model(input_shape=(64, 64, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name="feature_layer"),  # <== Key layer
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load model from file or create a new one
def load_cnn_model(model_path=None, input_shape=(64, 64, 1), num_classes=10):
    if model_path:
        try:
            model = models.load_model(model_path)
            return model
        except Exception as e:
            raise ValueError(f"Model loading failed: {e}")
    return create_cnn_model(input_shape, num_classes)

# Preprocess an image for prediction (grayscale expected)
def preprocess_image(image, target_size=(64, 64)):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert RGB to grayscale if necessary
    if image.ndim == 3 and image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)
    elif image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.ndim == 3 and image.shape[-1] == 1:
        pass  # Already grayscale
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    image = tf.image.resize(image, target_size)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)  # (1, H, W, C)
    return image

# Predict class index using the trained CNN
def predict_class(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    return predicted_class

# Build a feature extractor from the model by removing the softmax layer
# Build a feature extractor from the model by removing the softmax layer
def build_feature_extractor(model):
    feature_layer = None
    for layer in model.layers:
        if "feature" in layer.name:
            feature_layer = layer
            break

    if feature_layer is None:
        raise ValueError("No feature layer found in model")

    return models.Model(inputs=model.input, outputs=feature_layer.output)



# Extract deep features from an image using the CNN feature extractor
def extract_features(image, model):
    processed_image = preprocess_image(image)
    feature_model = build_feature_extractor(model)
    features = feature_model.predict(processed_image, verbose=0)
    return features.flatten()
