# Regression Bike Rental Prediction

This project demonstrates the application of a regression model using TensorFlow 2.0 to predict daily bike rental counts based on various factors such as weather conditions and weekdays.

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

The objective is to forecast daily bike rental demand based on historical usage patterns in relation to weather, environment, and other data.

## Dataset

The dataset contains daily records of bike-sharing counts along with corresponding weather and seasonal information. Key features include:

- **Temperature:** Average temperature in Celsius.
- **Humidity:** Average relative humidity.
- **Windspeed:** Average wind speed.
- **Season:** Categorical variable indicating the season (e.g., Spring, Summer).
- **Weekday:** Categorical variable indicating the day of the week.
- **Holiday:** Binary variable indicating whether the day is a holiday.
- **Working Day:** Binary variable indicating whether the day is a working day.

## Features

- **Data Visualization:** Explore data through various plots to understand trends and patterns.
- **Data Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features.
- **Model Building:** Construct a neural network using TensorFlow's Sequential API.
- **Model Training:** Train the model on the preprocessed data.
- **Model Evaluation:** Assess the model's performance using appropriate metrics.
- **Visualization:** Generate plots to visualize model performance and predictions.

## Model Architecture

The neural network model is built using TensorFlow's Sequential API and consists of the following layers:

1. **Input Layer:** Accepts the input features.
2. **Hidden Layers:** Multiple dense layers with ReLU activation functions.
3. **Output Layer:** A single neuron with a linear activation function to predict the bike rental count.

## Performance Metrics

The model's performance is evaluated using the following metrics:

- **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors between predicted and actual values.
- **R-squared (R²):** Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/virajindasrao/tensorflow2.0-bootcamp.git
   cd tensorflow2.0-bootcamp/01_regression_bike_rental_prediction


2. **Run Model:**
pip install -r requirements.txt

python 01_regression_bike_rental_prediction.py
