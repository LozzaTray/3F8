from plot_utils import plot_data
import numpy as np


def main():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Plotting data...")
    plot_data(X, y)


if __name__ == "__main__":
    main()