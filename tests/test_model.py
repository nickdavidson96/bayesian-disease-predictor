# tests/test_model.py

import torch
from src.model import BayesianNN

def test_model_forward():
    input_dim = 10
    model = BayesianNN(input_dim=input_dim)
    x = torch.randn(5, input_dim)
    output = model(x)
    assert output.shape == (5,), "Output shape mismatch"
    assert (output >= 0).all() and (output <= 1).all(), "Output not in [0, 1] range"