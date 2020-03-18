"""Module that contains plotting functions"""
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_predictive_general(X, y, predictor_func, map_inputs = lambda x : x, title=""):
    if title == "":
        title = "Predicitive Distribution"

    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predictor_func(X_tilde)
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


def plot_heatmap(matrix, xlabels, ylabels):
    ax = sns.heatmap(matrix, linewidth=0.5, xticklabels=xlabels, yticklabels=ylabels, annot=True)
    ax.set_xlabel("Initial variance (var_0)")
    ax.set_ylabel("RBF Width (l)")
    plt.show()


def display_confusion_array(conf):
    """
    Prints out normalised confusion matrix
    conf - array of error types [tn, fp, fn, tp]
    """
    true_zeroes = conf[0] + conf[1]
    true_ones = conf[2] + conf[3]

    conf[0] = conf[0] / true_zeroes
    conf[1] = conf[1] / true_zeroes

    conf[2] = conf[2] / true_ones
    conf[3] = conf[3] / true_ones

    print("Confusion array:")
    print("| tn | fp |\n| fn | tp |\n")
    print("| {} | {} |\n| {} | {} |".format(conf[0], conf[1], conf[2], conf[3]))