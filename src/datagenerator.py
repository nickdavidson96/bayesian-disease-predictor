import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 1000

# Generate features
age = np.random.randint(20, 90, size=n)
bmi = np.random.uniform(18, 40, size=n)
blood_pressure = np.random.uniform(90, 180, size=n)
glucose = np.random.uniform(70, 200, size=n)
cholesterol = np.random.uniform(120, 300, size=n)
smoker = np.random.binomial(1, 0.3, size=n)
physical_activity = np.random.uniform(0, 10, size=n)

# Weighted risk score
risk_score = (
    0.03 * age +
    0.1 * bmi +
    0.05 * blood_pressure +
    0.07 * glucose +
    0.04 * cholesterol +
    0.2 * smoker -
    0.1 * physical_activity
)

# Normalize and apply sigmoid
risk_prob = 1 / (1 + np.exp(-(risk_score - np.mean(risk_score)) / np.std(risk_score)))

# Binary label
disease_risk = np.random.binomial(1, risk_prob)

# Assemble DataFrame
df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "blood_pressure": blood_pressure,
    "glucose": glucose,
    "cholesterol": cholesterol,
    "smoker": smoker,
    "physical_activity": physical_activity,
    "disease_risk": disease_risk
})

# Save to CSV
df.to_csv(r"C:\Users\nicda\OneDrive\Documents\bayesian-disease-predictor\data\preprocessed\synthetic_patient_data.csv", index=False)

# Preview
df.head(10)
