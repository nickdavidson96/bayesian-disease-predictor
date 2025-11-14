# src/config.py

import os

# Paths
DATA_PATH = os.path.join("data", "processed", "patient_data.csv")
MODEL_SAVE_PATH = os.path.join("experiments", "run_001", "model.pt")

# Training
SEED = 42 
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
NUM_SAMPLES = 100

# Target
TARGET_COLUMN = "disease_risk"