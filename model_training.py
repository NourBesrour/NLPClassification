import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Load FastText embeddings
def load_fasttext_embeddings(file_path):
    model = FastText.load_fasttext_format(file_path)
    return model

# Load data
df = pd.read_csv('cleaned_data_utf8.csv') 
df = df[['text_cleaned', 'target']].dropna()
# shuffle the DataFrame rows
df = df.sample(frac = 1)
 

# Preprocess data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text_cleaned'])
sequences = tokenizer.texts_to_sequences(df['text_cleaned'])
X = pad_sequences(sequences, maxlen=100)
y = pd.get_dummies(df['target']).values  # One-hot encoding

vocab_size = len(tokenizer.word_index) + 1

# Load embeddings
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
fasttext_model = load_fasttext_embeddings('cc.en.300.bin')

# Create embedding matrices
embedding_dim_glove = 100
embedding_matrix_glove = np.zeros((vocab_size, embedding_dim_glove))

for word, i in tokenizer.word_index.items():
    if word in glove_embeddings:
        embedding_matrix_glove[i] = glove_embeddings[word]

embedding_dim_fasttext = 300
embedding_matrix_fasttext = np.zeros((vocab_size, embedding_dim_fasttext))

for word, i in tokenizer.word_index.items():
    if word in fasttext_model.wv:
        embedding_matrix_fasttext[i] = fasttext_model.wv[word]

# Define models
def create_bilstm_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model

def create_cnn_model(embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)
results_df = pd.DataFrame(columns=['Model', 'Embedding', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])


# Function to calculate and store metrics
def evaluate_model(model_name, embedding_type, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    # Append the results to the DataFrame
    results_df.loc[len(results_df)] = [model_name, embedding_type, accuracy, precision, recall, f1_score]
# Compile and train models with both embeddings

for embedding_matrix, embedding_type in zip([embedding_matrix_glove, embedding_matrix_fasttext], ['GloVe', 'FastText']):
    print(f"\nTraining models with {embedding_type} embeddings")

    # BiLSTM
    bilstm_model = create_bilstm_model(embedding_matrix)
    bilstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    bilstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate BiLSTM
    y_pred_bilstm = bilstm_model.predict(X_val)
    y_pred_bilstm_classes = np.argmax(y_pred_bilstm, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    evaluate_model("BiLSTM", embedding_type, y_val_classes, y_pred_bilstm_classes)


    # CNN
    cnn_model = create_cnn_model(embedding_matrix)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate CNN
    y_pred_cnn = cnn_model.predict(X_val)
    y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)
    evaluate_model("CNN", embedding_type, y_val_classes, y_pred_cnn_classes)

print("\nClassification Metrics for All Models:\n")
print(results_df)
# After training the GloVe CNN model:
cnn_model.save('glove_cnn_model.h5')


