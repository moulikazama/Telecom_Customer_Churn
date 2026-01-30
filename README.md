📊 Customer Churn Prediction using PyTorch
🔍 Project Overview

This project focuses on building an end-to-end customer churn prediction system using classical machine learning models and a Neural Network implemented in PyTorch. The goal is to identify customers who are likely to churn, with special emphasis on recall optimization, as missing churn customers can directly impact business revenue.

The project follows a complete real-world ML workflow: EDA → preprocessing → model building → evaluation → comparison.

📁 Dataset

Name: IBM Telco Customer Churn Dataset

Type: Real-world business dataset

Target Variable: Churn (Yes / No)

Problem Type: Binary Classification

This dataset includes customer demographics, account information, service usage, and billing details.

🧠 Key Concepts Covered

Exploratory Data Analysis (EDA)

Handling missing values and outliers

Encoding categorical variables (Ordinal & One-Hot Encoding)

Feature scaling (StandardScaler / MinMaxScaler)

Class imbalance handling

Threshold tuning

Model evaluation using confusion matrix, precision, recall, F1-score

Comparison of multiple models

🛠️ Tech Stack

Programming Language: Python

Libraries:

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

PyTorch

🤖 Models Implemented

Logistic Regression (Baseline model)

Decision Tree Classifier

Neural Network using PyTorch

Fully connected layers

ReLU activation

Binary Cross-Entropy Loss

Adam Optimizer

📈 Model Evaluation Metrics

Accuracy

Precision

Recall (Primary focus)

F1-Score

Confusion Matrix (TP, TN, FP, FN)

Recall is prioritized to ensure maximum identification of churn customers.

## PyTorch Neural Network vs Logistic Regression

- Implemented a PyTorch NN for churn prediction
- Compared with Logistic Regression baseline
- Evaluated with accuracy, precision, recall, and confusion matrix
- Observed NN has higher precision, Logistic has higher recall for churn

## 📊 Results Summary
- Logistic Regression: Baseline model with interpretable coefficients
- Decision Tree: Improved non-linear pattern capture
- PyTorch Neural Network: Best recall for churn customers after threshold tuning
