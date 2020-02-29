from plot_utils import plot_data, plot_ll, plot_predictive_distribution, display_confusion_array
from prob_utils import get_x_tilde, randomly_partition, fit_w, compute_confusion_array
import numpy as np


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

    return w, confusion, (ll_train[-1], ll_test[-1])


def expanded_main():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Randomly partitioning into training and test sets")
    n_train = 800
    X_train, y_train, X_test, y_test = randomly_partition(X, y, n_train)

    l = [0.01, 0.1, 1]
    a = [0.01, 0.01, 0.0001]
    n = [1000, 1000, 1000]

    for i in range(0, 3):
        expansion_function = lambda x: evaluate_basis_functions(l[i], x, X_train)
        w_exp, conf, ll_final = expanded_classifier(X_train, y_train, X_test, y_test, l=l[i], a=a[i], n=n[i])
        display_confusion_array(conf)
        plot_predictive_distribution(X, y, w_exp, expansion_function)


if __name__ == "__main__":
    print("Expanded classifiers:")
    expanded_main()