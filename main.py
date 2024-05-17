import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load and prepare the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((32, 32))
    img = np.array(img)
    if img.shape[-1] == 4:  # Check if the image has an alpha channel
        img = img[..., :3]  # Remove the alpha channel
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img /= 255.0
    return img

# Function to make predictions
def predict(image, model, labels):
    img = load_image(image)
    result = model.predict(img)
    predicted_class = np.argmax(result, axis=1)
    return labels[predicted_class[0]]

# Load the model
model = load_model('app.h5')  # Ensure the model file is named app.h5

# Function to load labels from a text file
def load_labels(filename):
    with open(filename, 'r') as file:
        labels = file.readlines()
    labels = [label.strip() for label in labels]
    return labels

# Streamlit UI
def main():
    # Sidebar
    st.sidebar.title("Group 1 Model Deployment in the Cloud")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

    if page == "Home":
        # Title
        st.title("Sports Prediction Model Application")

        # Main page content
        st.write("Welcome to the Sports Classification App! This app uses a Convolutional Neural Network (CNN) model to classify images into different sports categories.")
        st.write("Upload an image of a sports event or activity, and the app will predict whether it belongs to one of the following sports categories:")

        # List of sports categories
        sports_categories = [
            "Air Hockey", "Amputee Football", "Archery", "Arm Wrestling", "Axe Throwing", "Balance Beam",
            "Barrel Racing", "Baseball", "Basketball", "Baton Twirling", "Bike Polo", "Billiards", "BMX", "Bobsled",
            "Bowling", "Boxing", "Bull Riding", "Bungee Jumping", "Canoe Slalom", "Cheerleading", "Chuckwagon Racing",
            "Cricket", "Croquet", "Curling", "Disc Golf", "Fencing", "Field Hockey", "Figure Skating Men",
            "Figure Skating Pairs", "Figure Skating Women", "Fly Fishing", "Football", "Formula 1 Racing", "Frisbee",
            "Gaga", "Giant Slalom", "Golf", "Hammer Throw", "Hang Gliding", "Harness Racing", "High Jump", "Hockey",
            "Horse Jumping", "Horse Racing", "Horseshoe Pitching", "Hurdles", "Hydroplane Racing", "Ice Climbing",
            "Ice Yachting", "Jai Alai", "Javelin", "Jousting", "Judo", "Lacrosse", "Log Rolling", "Luge",
            "Motorcycle Racing", "Mushing", "NASCAR Racing", "Olympic Wrestling", "Parallel Bar", "Pole Climbing",
            "Pole Dancing", "Pole Vault", "Polo", "Pommel Horse", "Rings", "Rock Climbing", "Roller Derby",
            "Rollerblade Racing", "Rowing", "Rugby", "Sailboat Racing", "Shot Put", "Shuffleboard", "Sidecar Racing",
            "Ski Jumping", "Sky Surfing", "Skydiving", "Snowboarding", "Snowmobile Racing", "Speed Skating",
            "Steer Wrestling", "Sumo Wrestling", "Surfing", "Swimming", "Table Tennis", "Tennis", "Track Bicycle",
            "Trapeze", "Tug of War", "Ultimate", "Uneven Bars", "Volleyball", "Water Cycling", "Water Polo",
            "Weightlifting", "Wheelchair Basketball", "Wheelchair Racing", "Wingsuit Flying"
        ]

        st.write(sports_categories)

    elif page == "Prediction":
        # Prediction page
        st.title("Model Prediction")
        st.write("Upload an image to predict the sports category.")

        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            st.image(test_image, width=300, caption='Uploaded Image')
            if st.button("Predict"):
                st.write("Predicting...")
                labels = load_labels("labels.txt")
                predicted_sport = predict(test_image, model, labels)
                st.success(f"Predicted Sports Category: {predicted_sport}")

    elif page == "About":
        # About the project
        st.title("About the Project")
        st.write("""
        This Streamlit app uses a Convolutional Neural Network (CNN) model to classify sports categories. 
        It accepts an image as input and predicts whether the image belongs to one of the following sports categories: 
        Baseball, Basketball, Cricket, or Soccer.
        """)
        st.write("This project is based on a dataset available on Kaggle: [Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification/data)")
        st.write("Developed by:")
        st.write("- Altero, Alexia A.")
        st.write("- Balawang, Jhoana")
        st.write("- Bartolome, Ken Christian Adrian V.")
        st.write("- Berja, Christian Dave L.")
        st.write("- Corneta, Alfer Margaux")

if __name__ == "__main__":
    main()
