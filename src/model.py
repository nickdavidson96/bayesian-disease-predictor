# src/model.py

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianNN(PyroModule):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
         # Bayesian input layer
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        # Bayesian hidden layer
        self.fc2 = PyroModule[nn.Linear](hidden_dim, hidden_dim)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, hidden_dim]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))

        # Bayesian output layer
        self.out = PyroModule[nn.Linear](hidden_dim, 1)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.out(x)).squeeze(-1) # Output: probability of disease risk 