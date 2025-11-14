# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path, target_column, dropna=True):
    """
    Load dataset from CSV and return DataFrame.
    Args:
        path (str): Path to CSV file
        target_column (str): Name of target column
        dropna (bool): Whether to drop rows with missing values
    Returns:
    df (DataFrame): Cleaned DataFrame
    """
    df = pd.read_csv(path)
    if dropna:
        df = df.dropna()
    return df

def preprocess_data(df, target_column):
    """
    Split features and target, normalize features.
    Args:
        df (DataFrame): Input data
        target_column (str): Name of target column
    Returns:
        X_train, X_test, y_train, y_test: Split and scaled data
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def get_data(path, target_column):
    """
    Full pipeline: load, clean, split, scale
    """
    df = load_data(path, target_column)
    return preprocess_data(df, target_column)