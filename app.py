import streamlit as st
import requests
import torch

from transformers import BartTokenizer, BartForConditionalGeneration

# -------- PAGE CONFIG --------
st.set_page_config(page_title="AI News Summarizer", page_icon="📰")

st.title("News Summarizer")

st.write("Paste a news article and generate an AI summary")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():

    model_name = "facebook/bart-large-cnn"

    tokenizer = BartTokenizer.from_pretrained(model_name)

    model = BartForConditionalGeneration.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

# -------- USER INPUT --------
text = st.text_area("Paste news article text here", height=300)

# -------- SUMMARIZE FUNCTION --------
def summarize(text):

    inputs = tokenizer.encode(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=120,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# -------- BUTTON --------
if st.button("Generate Summary"):

    if text.strip() == "":
        st.warning("Please paste article text")
    else:

        with st.spinner("Generating AI summary..."):

            summary = summarize(text)

        st.subheader("Summary")

        st.success(summary)