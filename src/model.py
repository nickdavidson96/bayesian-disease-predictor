# src/model.py

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianNN(PyroModule):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 64)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([64, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([64]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](64, 32)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([32, 64]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([32]).to_event(1))

        self.out = PyroModule[nn.Linear](32, 1)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([1, 32]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))

        self.relu = nn.ReLU()

    def forward(self, x, y=None):  # Now properly indented inside the class
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x
