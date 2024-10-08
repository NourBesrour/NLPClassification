﻿# Text Classification with CNN (GloVe Embeddings)

This project implements a text classification application using a Convolutional Neural Network (CNN) model with GloVe embeddings. The application classifies sentences into five mental health categories: Stress, Depression, Bipolar Disorder, Personality Disorder, and Anxiety. It leverages natural language processing to analyze and classify text from the Reddit mental health dataset.

## Features

- **Model**: Utilizes a CNN model trained on GloVe embeddings for effective sentence classification.
- **Categories**: Predicts the following labels:
  - Stress
  - Depression
  - Bipolar Disorder
  - Personality Disorder
  - Anxiety
- **User Interface**: Built with Gradio, providing an intuitive web interface for input and output.
- **Hugging Face link** : https://huggingface.co/spaces/NourBesrour/NLPClassification/tree/main
## Installation

To run this application, ensure you have the following dependencies installed:

```bash
pip install numpy tensorflow gradio
