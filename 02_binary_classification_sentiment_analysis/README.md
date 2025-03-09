# Binary Classification: Sentiment Analysis

This project demonstrates the application of a binary classification model using TensorFlow 2.0 to perform sentiment analysis on textual data, determining whether a given text expresses a positive or negative sentiment.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [License](#license)

## Problem Statement

The objective is to classify textual data into positive or negative sentiment categories using a neural network model.

## Dataset

The dataset contains text samples labeled with their corresponding sentiments:

- **Text:** The textual data to be analyzed.
- **Sentiment:** Binary label indicating the sentiment (0 for negative, 1 for positive).

## Features

- **Data Preprocessing:** Tokenize and pad sequences, encode labels, and split data into training and testing sets.
- **Model Building:** Construct a neural network using TensorFlow's Sequential API.
- **Model Training:** Train the model on the preprocessed data.
- **Model Evaluation:** Assess the model's performance using appropriate metrics.
- **Visualization:** Generate plots to visualize model performance and predictions.

## Model Architecture

The neural network model is built using TensorFlow's Sequential API and consists of the following layers:

1. **Embedding Layer:** Converts input sequences into dense vectors of fixed size.
2. **LSTM Layer:** Captures temporal dependencies in the data.
3. **Dense Layers:** Fully connected layers with ReLU activation functions.
4. **Output Layer:** A single neuron with a sigmoid activation function to predict the probability of positive sentiment.

## Performance Metrics

The model's performance is evaluated using the following metrics:

- **Accuracy:** The proportion of correctly classified samples.
- **Precision:** The proportion of positive identifications that were actually correct.
- **Recall:** The proportion of actual positives that were correctly identified.
- **F1 Score:** The harmonic mean of precision and recall.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/virajindasrao/tensorflow2.0-bootcamp.git
   cd tensorflow2.0-bootcamp/02_binary_classification_sentiment_analysis

2. **Test Model:**
pip install -r requirements.txt

python 02_binary_classification_sentiment_analysis.py
