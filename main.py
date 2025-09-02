# main.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")

# --- Helper Functions ---

# Negation-aware preprocessing
def preprocess_text(text):
    words = text.lower().split()
    processed_words = []
    
    skip_next = False
    for i, word in enumerate(words):
        if word == "not" and i + 1 < len(words):
            # Merge "not" with the next word => "not_good"
            processed_words.append("not_" + words[i + 1])
            skip_next = True
        elif skip_next:
            skip_next = False
        else:
            processed_words.append(word)
    
    # Encode with IMDB word index (OOV → 2)
    encoded_review = [word_index.get(w, 2) + 3 for w in processed_words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# --- Streamlit UI ---

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("This app uses a **Simple RNN** model to classify movie reviews as **Positive** or **Negative**.")

# Sidebar examples
st.sidebar.header("Try Example Reviews")
examples = {
    "Positive Example": "The movie was absolutely wonderful with great acting and direction.",
    "Negative Example": "This was the worst film I have ever seen, complete waste of time.",
    "Negation Example": "movie is not good"
}
example_choice = st.sidebar.radio("Choose an example:", ["None"] + list(examples.keys()))

if example_choice != "None":
    user_input = examples[example_choice]
    st.info(f"Example selected: {user_input}")
else:
    user_input = st.text_area("✍️ Enter your movie review here:")

# Classify
if st.button("🔍 Classify Sentiment"):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = float(prediction[0][0])
        sentiment = "Positive 😀" if score > 0.5 else "Negative 😞"

        # Result display
        st.subheader("Prediction Result")
        st.success(f"Sentiment: **{sentiment}**")
        st.write(f"Prediction Score: `{score:.4f}` (higher = more positive)")

        # Progress bar
        st.progress(score if score > 0.5 else 1 - score)
    else:
        st.warning("Please enter a movie review before classifying.")
