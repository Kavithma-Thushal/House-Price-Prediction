import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


def generate_dataset():
    # Generate synthetic data for regression
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

    # Scale the target to represent house prices
    y = y * 1000

    # Plot the data
    plt.scatter(X, y, color='blue')
    plt.title("Synthetic House Prices Dataset")
    plt.xlabel("Size (in arbitrary units)")
    plt.ylabel("Price (in $)")
    plt.show()

    return X, y
