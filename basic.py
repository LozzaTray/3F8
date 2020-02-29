# This is the auxiliary code for the 3F8 coursework. Some parts are missing and
# should be completed by the student. These are Marked with XXX

import numpy as np
from plot_utils import plot_data, plot_ll, plot_predictive_distribution
from prob_utils import fit_w, predict, get_x_tilde


def compute_confusion_array(X_tilde, w, y):
    """computes confusion array"""
    output_probs = predict(X_tilde, w)
    assignments = (output_probs > 0.5)
    values = np.add(assignments, 2*y)

    types = [0, 0, 0, 0] # true negative = 0 , false positive = 1, false negative = 2, true positive = 3
    for i in range(0,4):
        types[i] = (values == i).sum()

    return types


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


def evaluate_basis_functions(l, X, Z):
    """
    Function that replaces initial input features by evaluating Gaussian basis functions
    on a grid of points

    Inputs:

    l: hyper-parameter for the width of the Gaussian basis functions
    Z: location of the Gaussian basis functions
    X: points at which to evaluate the basis functions

    Output: Feature matrix with the evaluations of the Gaussian basis functions.
    """
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)


def linear_classifier(X_train, y_train, X_test, y_test):
    """Performs tasks on the linear classifier and returns weights"""

    alpha = 0.001 # Learning rate
    n_steps = 100 # Number of steps

    X_tilde_train = get_x_tilde(X_train)
    X_tilde_test = get_x_tilde(X_test)

    print("Training linear classifier...")
    w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

    print("Plotting ll curve...")
    plot_ll(ll_train, ll_test, "Log-likelihood curve with linear classifier (alpha=" + str(alpha) + ")")

    print("ll_train | ll_test")
    print(str(ll_train[-1]) + " | " + str(ll_test[-1]))

    print("Computing linear classifier confusion on test data...")
    confusion = compute_confusion_array(X_tilde_test, w, y_test)

    return w, confusion


def expanded_classifier(X_train, y_train, X_test, y_test, l, a, n):

    print("Expanding data with basis functions")    
    l = l # Width of the Gaussian basis funcction
    X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
    X_tilde_test = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

    alpha = a # Learning rate
    n_steps = n # Number of steps

    print("Training expanded classifier...")
    w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

    print("Plotting ll curve...")
    plot_ll(ll_train, ll_test, 'Log-likelihood curve with expanded classifier (alpha=' + str(alpha) + ", l=" + str(l) + ")")

    print("ll_train | ll_test")
    print(str(ll_train[-1]) + " | " + str(ll_test[-1]))

    print("Computing classifier confusion on test data...")
    confusion = compute_confusion_array(X_tilde_test, w, y_test)

    return w, confusion


def randomly_partition(X, y, n_train):
    """randomly partitions into training and test sets"""
    # random permutation
    permutation = np.random.permutation(X.shape[ 0 ])
    X = X[ permutation, : ]
    y = y[ permutation ]

    # split into training and test sets
    n_train = 800
    X_train = X[ 0 : n_train, : ]
    X_test = X[ n_train :, : ]
    y_train = y[ 0 : n_train ]
    y_test = y[ n_train : ]

    return X_train, y_train, X_test, y_test


def main():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Randomly partitioning into training and test sets")
    n_train = 800
    X_train, y_train, X_test, y_test = randomly_partition(X, y, n_train)

    print("Plotting data basic...")
    plot_data(X, y)

    print("Linear classifier:")
    w_lin, conf = linear_classifier(X_train, y_train, X_test, y_test)
    display_confusion_array(conf)
    plot_predictive_distribution(X, y, w_lin)

    print("Expanded classifiers:")
    l = [0.01, 0.1, 1]
    a = [0.01, 0.01, 0.0005]
    n = [1000, 1000, 1000]

    for i in range(0, 3):
        expansion_function = lambda x: evaluate_basis_functions(l[i], x, X_train)
        w_exp, conf = expanded_classifier(X_train, y_train, X_test, y_test, l=l[i], a=a[i], n=n[i])
        display_confusion_array(conf)
        plot_predictive_distribution(X, y, w_exp, expansion_function)


if __name__ == "__main__":
    main()