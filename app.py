"""
Streamlit demo app for the LSTM sentiment analysis model.

Usage:
    streamlit run app.py

Note: run `python main.py` at least once first so a trained model
exists at models/sentiment_lstm_model.h5.
"""

import os
import streamlit as st
import tensorflow as tf

from src import config
from src.data_loader import get_word_index
from src.predict import predict_sentiment
# Page setup
st.set_page_config(
    page_title="Sentiment Analysis - LSTM",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "An LSTM neural network trained on 25,000 IMDB movie reviews. "
    "Type a review below and the model will predict whether it's "
    "positive or negative."
)

# Load model + word index (cached so this runs only once, not on every interaction)
@st.cache_resource
def load_trained_model():
    if not os.path.exists(config.MODEL_PATH):
        return None
    return tf.keras.models.load_model(config.MODEL_PATH)


@st.cache_resource
def load_word_index():
    return get_word_index()


model = load_trained_model()
word_index = load_word_index()

if model is None:
    st.error(
        f"No trained model found at `{config.MODEL_PATH}`. "
        "Run `python main.py` first to train and save the model."
    )
    st.stop()

# Input + prediction
review_text = st.text_area(
    "Enter a movie review:",
    placeholder="e.g. This movie was a complete waste of time...",
    height=120,
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    if not review_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, prob = predict_sentiment(model, word_index, review_text)

        if label == "Positive":
            st.success(f"**Positive** 🎉  (confidence: {prob:.1%})")
        else:
            st.error(f"**Negative** 😞  (confidence: {1 - prob:.1%})")

        st.progress(float(prob))
        st.caption(f"Raw model output (probability of positive sentiment): {prob:.4f}")

# Sample reviews for quick testing
st.divider()
st.subheader("Try a sample review")

samples = {
    "Glowing review": "This movie was absolutely wonderful, the acting was superb and the story kept me engaged the whole time.",
    "Harsh review": "Terrible plot, wooden acting, and a complete waste of two hours. I want my money back.",
    "Mixed review": "It was an okay movie, nothing special but not bad either. Some parts dragged.",
}

sample_cols = st.columns(len(samples))
for col, (label, text) in zip(sample_cols, samples.items()):
    with col:
        if st.button(label, use_container_width=True):
            pred_label, prob = predict_sentiment(model, word_index, text)
            st.write(f"*\"{text}\"*")
            if pred_label == "Positive":
                st.success(f"**Positive** ({prob:.1%})")
            else:
                st.error(f"**Negative** ({1 - prob:.1%})")

st.divider()
st.caption(
    "Built with TensorFlow/Keras (LSTM) and Streamlit. "
    "[View source on GitHub](https://github.com/<your-username>/sentiment-analysis-lstm)"
)
