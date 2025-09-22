import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Heart Disease dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Heart Disease dataset.
        y (numpy.ndarray): The target values of the Heart Disease dataset.
    """
    heart_data = pd.read_csv(r'D:\MLOps\Lab1\MLOps\Labs\API_Labs\FastAPI_Labs\src\data\heart.csv')
    X = heart_data.drop('target', axis=1).values
    y = heart_data['target'].values
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test