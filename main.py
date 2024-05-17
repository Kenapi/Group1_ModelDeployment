import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("Sports Prediction Model Application")

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("Fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("Vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (345 images each)")
    st.text("2. test (37 images each)")
    st.text("3. validation (37 images each)")
    st.subheader("Members:")
    st.text("Guanizo, Rcel James")
    st.text("Lagao, Miles Joshua")
    st.text("Laxamana, Abigail")
    st.text("Loreno, Eric H.")
    st.text("Mallari, Ashley")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        if st.button("Predict"):
            st.success("Our Prediction")
            with open("temp_image.jpg", "wb") as f:
                f.write(test_image.getbuffer())
            result_index = model_prediction("temp_image.jpg")
            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]
            st.success("Model Prediction: {}".format(label[result_index]))
