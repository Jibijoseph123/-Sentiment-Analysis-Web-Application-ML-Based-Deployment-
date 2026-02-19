#Streamlit App
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def predict(text):
    vect = tfidf.transform([text])
    return model.predict(vect)[0]

st.title("Movie Review Sentiment Analysis")

user_text = st.text_area("Enter review here:")

if st.button("Predict"):
    if user_text.strip() != "":
        sentiment = predict(user_text.lower())
        if sentiment == "positive":
            st.success("🙂 Positive Review")
        else:
            st.error("😔 Negative Review")
    else:
        st.warning("Please enter some text.")
