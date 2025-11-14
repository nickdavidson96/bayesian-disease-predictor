# src/utils.py

import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_banner(msg):
     print("=" * 50)
     print(f"{msg}")
     print("=" * 50)