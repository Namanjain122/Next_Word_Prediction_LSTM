import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Compatibility workaround for older LSTM models
from keras.layers import LSTM

original_from_config = LSTM.from_config.__func__

@classmethod
def compatible_from_config(cls, config):
    config.pop("time_major", None)
    return original_from_config(cls, config)

LSTM.from_config = compatible_from_config

model = load_model(
    "models/next_word_lstm.h5",
    compile=False
)

with open('models/tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# Streamlit app
st.title("Next Word Prediction LSTM")
input_text=st.text_input("Enter the sequence of words","To be or not to")


if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next Word: {next_word}')
