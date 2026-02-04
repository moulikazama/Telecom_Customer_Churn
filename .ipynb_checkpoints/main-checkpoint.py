import pandas as pd
from src.linear_regression import run_logistic_regression
from src.NeuralNetwork import run_neural_network

df = pd.read_csv("data/processed_data.csv")

print("Running Logistic Regression...")
lr_results = run_logistic_regression(df)

print("\nRunning Neural Network...")
nn_results = run_neural_network(df)

with open("results/metrics.txt", "w") as f:
    f.write("LOGISTIC REGRESSION\n")
    f.write(f"Accuracy: {lr_results['accuracy']}\n")
    f.write(str(lr_results['confusion_matrix']) + "\n")
    f.write(lr_results['report'] + "\n\n")

    f.write("NEURAL NETWORK\n")
    f.write(f"Accuracy: {nn_results['accuracy']}\n")
    f.write(str(nn_results['confusion_matrix']) + "\n")
    f.write(nn_results['report'])

print("\nResults saved in results/metrics.txt")