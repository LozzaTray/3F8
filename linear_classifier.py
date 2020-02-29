from plot_utils import plot_data, plot_ll, plot_predictive_distribution, display_confusion_array
from prob_utils import get_x_tilde, randomly_partition, fit_w, compute_confusion_array
import numpy as np


def linear_classifier():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Randomly partitioning into training and test sets")
    n_train = 800
    X_train, y_train, X_test, y_test = randomly_partition(X, y, n_train)

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
    display_confusion_array(confusion)

    print("Plotting predicitive distribution")
    plot_predictive_distribution(X, y, w)


if __name__ == "__main__":
    print("Linear classifier:")
    linear_classifier()