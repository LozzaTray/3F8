import numpy as np
from utils import plot_data


def augment(X):
    n,m = np.size(X)
    ones = np.ones(n)
    Xtilde = np.concatenate(ones, X)
    return Xtilde


def read_and_plot():
    """Reads input data and plots it"""
    X = np.loadtxt("X.txt")
    y = np.loadtxt("y.txt")
    Xtilde = augment(X)
    plot_data(X, y)


if __name__ == "__main__":
    read_and_plot()