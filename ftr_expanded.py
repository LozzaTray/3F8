from plot_utils import plot_ll, plot_predictive_general, display_confusion_array
from prob_utils import get_x_tilde, randomly_partition, get_confusion_matrix, get_average_ll, predict
from expanded_classifier import evaluate_basis_functions
from bayesian import find_map
import numpy as np


def display_metrics(y_hat, y, message=""):
    print(message)
    print("########################")
    
    print("Confusion matrix:")
    conf_matrix = get_confusion_matrix(y_hat, y)
    print(conf_matrix)
    
    print("------------------------")

    print("Average Log-Likelihood:")
    ll = get_average_ll(y_hat, y)
    print(ll)

    print("########################\n")


def ftr():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Randomly partitioning into training and test sets")
    n_train = 800
    X_train, y_train, X_test, y_test = randomly_partition(X, y, n_train)

    l_arr = [0.1]
    var_0_arr = [1]

    for l in l_arr:
        for var_0 in var_0_arr:
            print("l={} var_0={}".format(l, var_0))
            X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
            X_tilde_test  = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

            print("Finding MAP on expanded inputs...")
            w_map, Z, predict_laplace = find_map(X_tilde_train, y_train, var_0)
            predict_map = lambda X: predict(X_tilde=X, w=w_map)

            probs_test_laplace = predict_laplace(X_tilde_test)
            display_metrics(probs_test_laplace, y_test, "Test laplace")
            
            probs_train_laplace = predict_laplace(X_tilde_train)
            display_metrics(probs_train_laplace, y_train, "Train laplace")

            probs_test_map = predict_map(X_tilde_test)
            display_metrics(probs_test_map, y_test, "Test map")

            probs_train_map = predict_map(X_tilde_train) 
            display_metrics(probs_train_map, y_train, "Train map")

            expansion_function = lambda x: evaluate_basis_functions(l, x, X_train)
            plot_predictive_general(X, y, predict_laplace, expansion_function)


if __name__ == "__main__":
    print("FTR")
    ftr()