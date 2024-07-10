# Premier League Match Predictor

## Overview

This project uses machine learning to predict the outcomes of Premier League football matches. It employs a Random Forest Classifier to analyze historical match data and forecast future results. The data used spans from 2020 to 2022, and various features including categorical, temporal, and rolling averages of key performance metrics are used to enhance the prediction accuracy.

## Features

- **Random Forest Classifier**: Utilizes scikit-learn to implement a Random Forest Classifier for predicting match outcomes.
- **Data Preprocessing**: Transforms raw match data, including conversion of categorical and temporal features, to prepare it for model training.
- **Rolling Averages**: Integrates rolling averages of critical performance metrics to account for team form and improve prediction capabilities.
- **Model Evaluation**: Performs hyperparameter tuning and cross-validation to fine-tune model parameters and evaluate performance using accuracy, precision, recall, and f1-score metrics.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/pl-match-predictor.git
    cd pl-match-predictor
    ```

2. **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn
    ```

3. **Place the dataset (`matches.csv`) in the project directory.**

## Usage

1. **Run the Predictor script:**
    ```bash
    python Predictor.py
    ```

2. **Script Output:**
    - **Cross-validation scores:** Provides precision scores for each fold.
    - **Accuracy and Precision:** Displays overall accuracy and precision of the model.
    - **Confusion Matrix and Classification Report:** Offers a detailed breakdown of the model’s performance.
    - **Combined and Merged Results DataFrames:** Shows actual versus predicted results and detailed match data.

## Example Output

```text
Cross-validation precision scores: [0.37777778 0.48275862 0.4939759  0.63888889 0.42424242]
Mean cross-validation precision: 0.4835287230426408
Accuracy: 0.6666666666666666
Precision: 0.6153846153846154
Confusion Matrix:
[[152  20]
 [ 72  32]]
Classification Report:
              precision    recall  f1-score   support

           0       0.68      0.88      0.77       172
           1       0.62      0.31      0.41       104

    accuracy                           0.67       276
   macro avg       0.65      0.60      0.59       276
weighted avg       0.65      0.67      0.63       276
