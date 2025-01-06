from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('lstm_model.h5')  # Adjust this path if needed

# Load your tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max_sequence_len based on your preprocessing
max_sequence_len = 10  # Adjust according to your preprocessing setup

def predict_next_words(model, tokenizer, seed_text, max_sequence_len, next_words, top_n=5):
    generated_words = []
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        top_n_indices = np.argsort(predicted_probs)[-top_n:][::-1]
        word_options = [word for word, index in tokenizer.word_index.items() if index in top_n_indices]
        
        if word_options:  # Check if there are any word options available
            next_word = word_options[0]  # Choose the top option for simplicity
            generated_words.append(next_word)
            seed_text += " " + next_word  # Append the predicted word to the seed text
    return generated_words

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    seed_text = data['seed_text']
    next_words = int(data.get('next_words', 1))  # Get number of words to predict
    top_n = int(data.get('top_n', 5))

    # Get next word options
    word_options = predict_next_words(model, tokenizer, seed_text, max_sequence_len, next_words, top_n)
    
    return jsonify({'next_word_options': word_options})

if __name__ == '__main__':
    app.run(debug=True)
