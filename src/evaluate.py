# src/evaluate.py

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from src.model import BayesianNN
from src.data_loader import get_data

def predict(model, x_test, num_samples=100):
    """
    Run multiple forward passes to capture uncertainty.
    """
    model.eval()
    preds = []
    for _ in range(num_samples):
        with torch.no_grad():
            output = model(toch.tensor(x_test, dtype=torch.float32))
            preds.append(output.numpy())
    return preds

def evaluate(x_test, y_test, model):
    """
    Evaluate model performance and visualize uncertainty
    """
    preds = predict(model, x_test)
    mean_preds = torch.tensor(preds).mean(dim=0).numpy()

    acc = accuracy_score(y_test, mean_preds > 0.5)
    auc = roc_auc_score(y_test, mean_preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC_AUC: {auc:.4f}")

    # Plot uncertainty
    plt.figure(figsize=(10, 5))
    plt.hist(torch.tensor(preds).std(dim=0).numpy(), bins=30, alpha=0.7)
    plt.title("Prediction Uncertainty (Std Dev)")
    plt.xlabel("Std Dev")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def main():
    X_train, X_test, y_train, y_test = get_data("data/preprocessed/patient_data.csv", target_column="disease_risk")
    model = BayesianNN(input_dim=X_test.shape[1])
    evaluate(X_test, y_test, model)

if __name__ == "__main__":
    main()