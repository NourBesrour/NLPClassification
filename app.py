import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import gradio as gr

# Load the pre-trained CNN model
model = load_model('glove_cnn_model.h5')


# Load GloVe embeddings (you may not need to load them here if your model is already trained with GloVe)
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# Assuming your tokenizer has already been fit on your training data.
# You can either load the tokenizer from a file if saved earlier or reinitialize as done here.
tokenizer = Tokenizer()


# Optionally, you may want to load pre-saved tokenizer (use tokenizer = load_tokenizer(...))

# Preprocess input text
def preprocess_input_sentence(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    return padded_sequence


# Function to classify a sentence
def classify_sentence(sentence):
    processed_sentence = preprocess_input_sentence(sentence)
    prediction = model.predict(processed_sentence)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Example label mapping based on your categories
    label_mapping = {0: 'Stress', 1: 'Depression', 2: 'Bipolar disorder', 3: 'Personality disorder', 4: 'Anxiety'}

    return label_mapping[predicted_class]


def classify_with_gradio(sentence):
    predicted_label = classify_sentence(sentence)
    return f"The sentence is classified as: {predicted_label}"


gr_interface = gr.Interface(
    fn=classify_with_gradio,
    inputs="text",
    outputs="text",
    title="Text Classification with CNN (GloVe Embeddings)",
    description="Enter a sentence to classify it using a CNN model trained on GloVe embeddings."
)

if __name__ == "__main__":
    gr_interface.launch(share=True)
