import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def explain_with_shap(model, X_test):
    X_np = np.array(X_test)

    # Wrapper to convert NumPy input to tensor
    def model_wrapper(x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            return model(x_tensor).numpy()

    masker = shap.maskers.Independent(X_np)
    explainer = shap.Explainer(model_wrapper, masker)
    shap_values = explainer(X_np)
    shap.plots.beeswarm(shap_values)
