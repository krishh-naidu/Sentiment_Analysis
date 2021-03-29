import streamlit as st
import pandas as pd 
from transformers import pipeline

'''
# Sentiment Analysis Tool

Please Enter something in Text Area to find out the sentiment
'''

@st.cache()
def analysis(text):
    classifier = pipeline('sentiment-analysis')
    return classifier(text)


text = st.text_area('Enter what you think')

if st.button('Check Sentiment'):
    if not text:
        st.warning('Input something')
    else:
        res = analysis(text)
        label = res[0]['label']
        score = "{:.2%}".format(res[0]['score'])
        st.success(f'Model characterizes the comment as {label} with confidence level of {score}')