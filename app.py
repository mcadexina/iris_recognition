import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.preprocessing import preprocess_image  # Assuming this function is already defined
from models.cnn_model import load_cnn_model, extract_features, predict_class  # Corrected CNN Model functions
from models.gabor_model import extract_features as gabor_features  # Gabor filter model
from models.wavelet_model import extract_features as wavelet_features  # Wavelet model

# Hardcoded username and password
USERNAME = "admin"
PASSWORD = "mypassword"

# Function for login
def login():
    st.title("Iris Recognition System Login")
    
    # User input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True  # Set session variable
            st.success("Login successful! Redirecting to the dashboard...")
            st.rerun()  # Refresh to show dashboard (using the updated method)
        else:
            st.error("Invalid username or password. Please try again.")

# Dashboard function
def dashboard():
    st.title("Iris Recognition Dashboard")

    st.subheader("Upload Iris Image for Recognition")
    uploaded_file = st.file_uploader("Choose an iris image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Iris Image", use_container_width=True)

        # Convert uploaded file to RGB NumPy array
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Preprocess
        processed_img = preprocess_image(image_np)
        st.success("Image successfully preprocessed ✅")

        # 🌟 Model selection
        st.subheader("Select Feature Extraction Method")
        model_choice = st.selectbox("Choose a model:", ["Gabor Filter", "2D Complex Wavelet Transform", "CNN"])

        if st.button("Extract Features"):
            if model_choice == "Gabor Filter":
                features = gabor_features(processed_img)
                st.success("Features extracted using Gabor Filter ✅")

            elif model_choice == "2D Complex Wavelet Transform":
                features = wavelet_features(processed_img)
                st.success("Features extracted using 2D Complex Wavelet Transform ✅")

            elif model_choice == "CNN":
                model = load_cnn_model(model_path="iris_cnn_model.h5")
                st.success("CNN model loaded ✅")

                features = extract_features(image_np, model)
                st.success("Features extracted using CNN ✅")

                prediction = predict_class(image_np, model)
                st.write(f"Predicted Class: {prediction}")

            else:
                st.error("Invalid model selected.")
                return

            st.write(f"Feature vector length: {len(features)}")
            st.write("Feature sample:", features[:10])

# Main app logic
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        dashboard()
    else:
        login()

# Run the app
if __name__ == "__main__":
    main()
