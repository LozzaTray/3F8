"""Module that contains plotting functions"""
import matplotlib.pyplot as plt
import numpy as np
from prob_utils import get_x_tilde, predict


def plot_data_internal(X, y):
    """
    Function that plots the points in 2D together with their labels

    Inputs:

    X: 2d array with the input features
    y: 1d array with the class labels (0 or 1)

    Output: 2D matrices with the x and y coordinates of the points shown in the plot
    """

    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy


def plot_data(X, y):
    """
    Function that plots the data without returning anything by calling "plot_data_internal".

    Input:

    X: 2d array with the input features
    y: 1d array with the class labels (0 or 1)

    Output: Nothing.
    """
    plot_data_internal(X, y)
    plt.show()


def plot_predictive_distribution(X, y, w, map_inputs = lambda x : x, title=""):
    """
    Function that plots the predictive probabilities of the logistic classifier

    Input:

    X: 2d array with the input features for the data (without adding a constant column with ones at the beginning)
    y: 1d array with the class labels (0 or 1) for the data
    w: parameter vector
    map_inputs: function that expands the original 2D inputs using basis functions.

    Output: Nothing.
    """
    if title == "":
        title = "Predicitive Distribution"

    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict(X_tilde, w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.title(title)
    plt.show()


def plot_ll(ll_training, ll_test, title=''):
    """
    Function that plots the average log-likelihood returned by "fit_w"

    Input:

    ll: vector with log-likelihood values

    Output: Nothing
    """
    if title == '':
        title = 'Plot of Average Log-Likelihood Curve'
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll_training) + 2)
    plt.ylim(min(ll_training) - 0.1, max(ll_training) + 0.1)
    ax.plot(np.arange(1, len(ll_training) + 1), ll_training, 'r-', label="Training data")
    ax.plot(np.arange(1, len(ll_test) + 1), ll_test, 'b-', label="Test data")
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title(title)
    plt.legend()
    plt.show()

