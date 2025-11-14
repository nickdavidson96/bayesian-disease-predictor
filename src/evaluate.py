# src/evaluate.py

from pyro.infer import Predictive
from pyro.infer.autoguide import AutoNormal
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from src.model import BayesianNN
from src.data_loader import get_data
from src.train import model_fn

def predict_with_uncertainty(model, guide, x_data, num_samples=1000):
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    predictive = Predictive(model_fn, guide=guide, num_samples=num_samples, return_sites=["obs"])
    samples = predictive(x_tensor, None)  # pass both x and y as in model_fn
    preds = samples["obs"]
    mean = preds.mean(dim=0).numpy()
    std = preds.std(dim=0).numpy()
    return mean, std
def predict(model, x_test, num_samples=100):
    """
    Run multiple forward passes to capture uncertainty.
    """
    model.eval()
    preds = []
    for _ in range(num_samples):
        with torch.no_grad():
            output = model(torch.tensor(x_test, dtype=torch.float32))
            preds.append(output.numpy())
    return preds

def evaluate(x_test, y_test, model, guide):
    """
    Evaluate model performance and visualize uncertainty
    """
    preds = predict(model, x_test)
    mean_preds = torch.tensor(np.array(preds)).mean(dim=0).numpy()

    # Posterior predictive sampling
    bayes_mean, bayes_std = predict_with_uncertainty(model, guide, x_test)

    # Metrics
    acc = accuracy_score(y_test, bayes_mean > 0.5)
    auc = roc_auc_score(y_test, bayes_mean)

    print(f"Accuracy (Bayesian): {acc:.4f}")
    print(f"ROC_AUC (Bayesian): {auc:.4f}")

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
    guide = AutoNormal(model)
    svi = SVI(model_fn, guide, optimizer, loss)

    for step in range(num_steps):
        loss = svi.step(x_tensor, y_tensor)
    evaluate(X_test, y_test, model, guide)

if __name__ == "__main__":
    main()