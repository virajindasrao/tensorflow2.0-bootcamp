# TensorFlow 2.0 Bootcamp

This repository offers hands-on projects to master TensorFlow 2.0, focusing on:

1. **Regression:** Predicting bike rental counts based on environmental and seasonal factors.
2. **Binary Classification:** Performing sentiment analysis to classify text as positive or negative.

## Table of Contents

- [Projects Overview](#projects-overview)
  - [1. Regression: Bike Rental Prediction](#1-regression-bike-rental-prediction)
  - [2. Binary Classification: Sentiment Analysis](#2-binary-classification-sentiment-analysis)
- [Setup Instructions](#setup-instructions)
- [License](#license)

## Projects Overview

### 1. Regression: Bike Rental Prediction

**Objective:** Forecast daily bike rental demand using historical data, considering factors like temperature, humidity, and wind speed.

**Dataset:** Contains daily records of bike-sharing counts with corresponding weather and seasonal information.

**Key Features:**

- **Data Visualization:** Explore trends and patterns in bike rentals.
- **Data Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features.
- **Model Building:** Construct a neural network using TensorFlow's Sequential API.
- **Model Training:** Train the model on preprocessed data.
- **Model Evaluation:** Assess performance using metrics like RMSE and R².
- **Visualization:** Generate plots to visualize model performance and predictions.

**Model Architecture:**

1. **Input Layer:** Accepts input features.
2. **Hidden Layers:** Multiple dense layers with ReLU activation functions.
3. **Output Layer:** Single neuron with a linear activation function to predict bike rental counts.

**Performance Metrics:**

- **Root Mean Squared Error (RMSE):** Measures average magnitude of errors between predicted and actual values.
- **R-squared (R²):** Indicates proportion of variance in the dependent variable predictable from independent variables.

**Reference:** For a similar approach, refer to this [project on predicting bike rentals](https://github.com/ZubairKhan87/bike-rental-demand-prediction).

### 2. Binary Classification: Sentiment Analysis

**Objective:** Classify textual data into positive or negative sentiment categories using a neural network model.

**Dataset:** Contains text samples labeled with corresponding sentiments (0 for negative, 1 for positive).

**Key Features:**

- **Data Preprocessing:** Tokenize and pad sequences, encode labels, and split data into training and testing sets.
- **Model Building:** Construct a neural network using TensorFlow's Sequential API.
- **Model Training:** Train the model on preprocessed data.
- **Model Evaluation:** Assess performance using metrics like accuracy, precision, recall, and F1 score.
- **Visualization:** Generate plots to visualize model performance and predictions.

**Model Architecture:**

1. **Embedding Layer:** Converts input sequences into dense vectors of fixed size.
2. **LSTM Layer:** Captures temporal dependencies in the data.
3. **Dense Layers:** Fully connected layers with ReLU activation functions.
4. **Output Layer:** Single neuron with a sigmoid activation function to predict probability of positive sentiment.

**Performance Metrics:**

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Proportion of positive identifications that were correct.
- **Recall:** Proportion of actual positives correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.

**Reference:** For a comprehensive guide on binary classification with TensorFlow, see this [tutorial](https://www.freecodecamp.org/news/binary-classification-made-simple-with-tensorflow/).

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/virajindasrao/tensorflow2.0-bootcamp.git
   cd tensorflow2.0-bootcamp
