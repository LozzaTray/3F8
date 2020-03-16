from plot_utils import plot_ll, plot_predictive_general, display_confusion_array
from prob_utils import get_x_tilde, randomly_partition, get_confusion_matrix
from expanded_classifier import evaluate_basis_functions
from bayesian import find_map
import numpy as np


def ftr():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Randomly partitioning into training and test sets")
    n_train = 800
    X_train, y_train, X_test, y_test = randomly_partition(X, y, n_train)

    l_arr = [0.1]
    var_0_arr = [1]

    for i in range(0, len(l_arr)):
        l = l_arr[i]
        for j in range(0, len(var_0_arr)):
            var_0 = var_0_arr[j]

            print("Finding MAP in linear case...")
            X_tilde_train_lin = get_x_tilde(X_train)
            X_tilde_test_lin = get_x_tilde(X_test)
            w_map_lin, Z_lin, predictor_func_lin = find_map(X_tilde_train_lin, y_train, var_0)
            
            probs_test = predictor_func_lin(X_tilde_test_lin)
            conf_matrix = get_confusion_matrix(probs_test, y_test)
            print("Linear confusion matrix")
            print(conf_matrix)

            plot_predictive_general(X, y, predictor_func_lin)

            print("Expanding inputs l={}".format(l))
            X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
            X_tilde_test  = get_x_tilde(evaluate_basis_functions(l, X_test, X_train))

            print("Finding MAP on expanded inputs...")
            w_map, Z, predictor_func = find_map(X_tilde_train, y_train, var_0)
            expansion_function = lambda x: evaluate_basis_functions(l, x, X_train)
            plot_predictive_general(X, y, predictor_func, expansion_function)


if __name__ == "__main__":
    print("FTR")
    ftr()