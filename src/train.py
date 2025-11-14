# src/train.py

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.model import BayesianNN
from src.data_loader import get_data

def model_fn(x_data, y_data=None):
    model = BayesianNN(input_dim=x_data.shape[1])
    with pyro.plate("data", x_data.shape[0]):
        output = model(x_data)
        pyron.sample("obs", dist.Bernoulli(probs=output), obs=y_data)
    return output

def guide_fn(x_data, y_data=None):
    model = BayesianNN(input_dim=x_data.shape[1])
    return model(x_data)

def train(x_train, y_train, num_steps=1000, lr=1e-3):
    pyro.clear_param_store()
    svi = SVI(model_fn, guide_fn, Adam({"lr": lr}), loss=Trace_ELBO())

    for step in range (num_steps):
        loss = svi.step(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        if step % 100 == 0:
            print(f"[step {step}] loss: {loss:.4f}")
def main():

    # Load and Preprocess data
    X_train, X_test, y_train, y_test = get_data("data/processed/patient_data.csv", target_column="disease_risk")

    # Train model
    train(X_train, y_train)

if __name__ == "__main__":
    main()