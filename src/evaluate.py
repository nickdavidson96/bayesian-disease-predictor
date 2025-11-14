# src/evaluate.py

from pyro.infer import Predictive
from pyro.infer.autoguide import AutoNormal
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
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

    sorted_indices = np.argsort(bayes_mean)
    sorted_mean = bayes_mean[sorted_indices]
    sorted_std = bayes_std[sorted_indices]
    sorted_labels = np.array(y_test)[sorted_indices]

    # Plot prediction intervals
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_mean, label="Mean Prediction", color="blue")
    plt.fill_between(
        range(len(sorted_mean)),
        sorted_mean - sorted_std,
        sorted_mean + sorted_std,
        color="blue",
        alpha=0.3,
        label="±1 Std Dev"
    )
    plt.plot(sorted_labels, label="True Labels", color="green", linestyle="--", alpha=0.6)
    plt.title("Prediction Intervals with True Labels")
    plt.xlabel("Sorted Sample Index")
    plt.ylabel("Predicted Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Metrics
    acc = accuracy_score(y_test, bayes_mean > 0.5)
    auc = roc_auc_score(y_test, bayes_mean)
    brier = brier_score_loss(y_test, bayes_mean)

    print(f"Accuracy (Bayesian): {acc:.4f}")
    print(f"ROC_AUC (Bayesian): {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # Plot uncertainty
    plt.figure(figsize=(10, 5))
    plt.hist(torch.tensor(preds).std(dim=0).numpy(), bins=30, alpha=0.7)
    plt.title("Prediction Uncertainty (Std Dev)")
    plt.xlabel("Std Dev")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    prob_true, prob_pred = calibration_curve(y_test, bayes_mean, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Bayesian Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame({
        "mean_pred":bayes_mean,
        "std_pred": bayes_std,
        "true_label": y_test
    })
    df.to_csv("outputs/predictions_with_uncertainty.csv", index=False)

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