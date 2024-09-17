import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and vectorizer
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model()

# Streamlit app title
st.title("Spam Mail Detection")

# Description for the app
st.write("""
This application uses a Machine Learning model to detect whether an email is spam or ham. 
Just input the text of your email, and it will predict the result.
""")

# Input text area for the email
input_email = st.text_area("Enter the email text below:")

# Predict button
if st.button("Predict"):
    if input_email:
        # Transform the input email using the vectorizer
        input_features = vectorizer.transform([input_email])

        # Predict using the loaded model
        prediction = model.predict(input_features)

        # Output the result
        if prediction[0] == 1:
            st.success("The email is predicted to be: Ham")
        else:
            st.error("The email is predicted to be: Spam")
    else:
        st.warning("Please enter some text to predict.")

# Footer
import streamlit as st


st.write("Model: Logistic Regression | Vectorization: TF-IDF")
