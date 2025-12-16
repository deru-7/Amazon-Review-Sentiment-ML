import streamlit as st
import pandas as pd
import pickle
import re

with open('vectorizer.pkl','rb') as file:
    vectorizer=pickle.load(file)

with open('model.pkl','rb') as file:
    model=pickle.load(file)


def clean_text(text:str)->str:
    text=text.lower()
    text=re.sub(r"[^a-z\s]","",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

label_map={
    0:'Negative',
    1:'Neutral',
    2:'Positive'
}

st.set_page_config(page_title="Amazon Review Sentiment",layout='centered')

st.title("Amazon Review Sentiment Analyzer")
st.write("Classifies Review into **Negative**,**Neutral**,**Positive**.")

review=st.text_area("Enter an Amazon product review",height=150)

if st.button("Analyze Sentiment"):
    if review.strip()=="":
        st.error("Empty input. Type a review.")
    else:
        cleaned=clean_text(review)
        vect_review=vectorizer.transform([cleaned])
        prediction=model.predict(vect_review)[0]
        sentiment=label_map[prediction]

        st.subheader("Prediction")
        if sentiment=="Negative":
            st.error(sentiment)
        elif sentiment=='Netural':
            st.warning(sentiment)
        else:
            st.success(sentiment)


st.markdown("---")
st.caption("ML-based sentiment analysis using TF-IDF + Linear Models")
