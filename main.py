# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model("simple_RNN_IMDB.h5")

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
MAX_FEATURES = 1000
MAX_LEN = 500

def preprocess_text(text):
    words = text.lower().split()

    encoded_review = []

    for word in words:
        # Get the original IMDB word index
        index = word_index.get(word)

        # Unknown word
        if index is None:
            encoded_review.append(2)
            continue

        # Add offset used by IMDB
        index = index + 3

        # Replace words outside vocabulary with OOV token
        if index >= MAX_FEATURES:
            index = 2

        encoded_review.append(index)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    return padded_review

import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

