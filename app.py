import streamlit as st
from src.inference import predict_sentiment

st.title("Review Sentiment Classification")
review = st.text_area("Enter review body:", height=200)


if st.button("Calssify Review"):
    percentage, sentiment = predict_sentiment(review)
    percentage = 100 - percentage if percentage<50 else percentage
    st.write(f"I'm {percentage:.2f}% confident that the review is {sentiment}")