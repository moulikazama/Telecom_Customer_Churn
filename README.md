
# 📊 Customer Churn Prediction using PyTorch & Logistic Regression

## 📌 Project Overview
Customer churn is a critical business problem for telecom companies, as retaining existing customers is more cost-effective than acquiring new ones.
This project focuses on predicting whether a customer will churn using historical telecom customer data.

Two models were built and compared:
- Logistic Regression (baseline ML model)
- Neural Network using PyTorch

The goal is to evaluate model performance and understand key factors influencing churn.

---

## 🧠 Problem Statement
Predict whether a customer will churn (Yes/No) based on demographic details, account information, and service usage patterns.

---

## 📂 Project Structure
```
customer-churn-pytorch/
│
├── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│
├── src/
│   ├── linear_regression.py
│   ├── NeuralNetwork.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── metrics.txt
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 🔍 Exploratory Data Analysis (EDA)
- Analyzed churn distribution and class imbalance
- Identified numerical vs categorical features
- Observed higher churn in month-to-month contracts and short-tenure customers
- Checked missing values and data consistency

---

## 🛠 Data Preprocessing
- Converted target variable (Churn) to binary (0/1)
- Handled missing values
- One-hot encoded categorical variables
- Scaled numerical features using StandardScaler
- Addressed class imbalance using class weights

---

## 🤖 Models Used

### Logistic Regression
- Baseline classification model
- Used class weighting to handle imbalance

### Neural Network (PyTorch)
- Fully connected feedforward neural network
- Binary classification using BCEWithLogitsLoss

---

## 📈 Model Evaluation

Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Results Summary
| Model | Accuracy |
|------|----------|
| Logistic Regression | ~75% |
| PyTorch Neural Network | ~77% |

---

## 🚀 How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run notebooks for EDA and preprocessing

4. Run models
```bash
python main.py
```

---

## 🧰 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch
- Matplotlib, Seaborn

---

## 📌 Key Learnings
- End-to-end ML workflow
- Handling class imbalance
- Comparing ML and Deep Learning models
- Structuring ML projects for GitHub

---

## 📬 Author
Mouli Shankar  
Aspiring Data Scientist | Machine Learning | Analytics
