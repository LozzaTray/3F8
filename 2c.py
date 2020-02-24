import numpy as np
from utils import plot_data


def read_and_plot():
    """Reads input data and plots it"""
    X = np.loadtxt("X.txt")
    y = np.loadtxt("y.txt")
    plot_data(X, y)


if __name__ == "__main__":
    read_and_plot()