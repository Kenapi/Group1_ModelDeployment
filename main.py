import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Function to load and prepare the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((32, 32))  # Adjust this size according to your model's input shape
    img = np.array(img)
    img = img.reshape(1, 32, 32, 3)  # Adjust this shape according to your model's input shape
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def predict(image, model, labels):
    img = load_image(image)
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=1)
    return labels[predicted_class[0]]

# Load the model with error handling
model_path = 'app.h5'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct directory.")
else:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Function to load labels from a text file
def load_labels(filename):
    with open(filename, 'r') as file:
        labels = file.readlines()
    labels = [label.strip() for label in labels]
    return labels

# Streamlit UI
def main():
    # Title
    st.title("Sports Prediction Model Application")

    # Main page content
    st.write("Welcome to the Sports Classification App! This app uses a Convolutional Neural Network (CNN) model to classify images into different sports categories.")
    st.write("Upload an image of a sports event or activity, and the app will predict the corresponding sports category.")

    # Prediction section
    st.header("Make a Prediction")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        if st.button('Predict'):
            st.write("Predicting...")
            labels = load_labels("labels.txt")  # Ensure labels.txt contains your class labels
            try:
                predicted_sport = predict(uploaded_image, model, labels)
                st.success(f"Predicted Sports Category: {predicted_sport}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
