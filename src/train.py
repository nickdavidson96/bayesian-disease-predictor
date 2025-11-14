# src/train.py

import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.model import BayesianNN

from src.data_loader import get_data

def model_fn(x_data, y_data=None):
    model = BayesianNN(input_dim=x_data.shape[1])
    output = model(x_data).squeeze(-1)  # shape: [batch_size]

    with pyro.plate("data", x_data.shape[0]):
        pyro.sample("obs", dist.Bernoulli(probs=output), obs=y_data)

def guide_fn(x_data, y_data=None):
    model = BayesianNN(input_dim=x_data.shape[1])
    guide = AutoNormal(model)
    return guide(x_data, y_data)
def train(x_train, y_train, num_steps=1000, lr=1e-3):
    pyro.clear_param_store()
    svi = SVI(model_fn, guide_fn, Adam({"lr": lr}), loss=Trace_ELBO())

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    for step in range(num_steps):
        loss = svi.step(x_tensor, y_tensor)
        if step % 100 == 0:
            print(f"[step {step}] loss: {loss:.4f}")
