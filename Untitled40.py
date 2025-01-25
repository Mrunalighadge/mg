

import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit application
st.title("Fake News Detection")
st.write("Enter a news article below to check if it's real or fake.")

# Input area
news_text = st.text_area("News Article:", placeholder="Type your news article here...")

# Predict button
if st.button("Predict"):
    if news_text.strip():
        # Vectorize input text and make prediction
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        result = "Real News" if prediction[0] == 1 else "Fake News"
        st.success(f"Prediction: {result}")
    else:
        st.error("Please enter a valid news article.")




