from plot_utils import plot_data, plot_ll, plot_predictive_distribution, display_confusion_array
from prob_utils import get_x_tilde, randomly_partition, fit_w, compute_confusion_array, compute_average_ll
from expanded_classifier import evaluate_basis_functions
from bayesian import find_map_and_hessian
from functools import partial
import numpy as np


def ftr():
    print("Loading data...")
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')

    print("Randomly partitioning into training and test sets")
    n_train = 800
    X_train, y_train, X_test, y_test = randomly_partition(X, y, n_train)

    l = [0.1]
    a = [0.01]
    n = [1000]
    sigma_0_squared = [1]

    for i in range(0, len(l)):
        X_tilde_train = get_x_tilde(evaluate_basis_functions(l[i], X_train, X_train))
        X_tilde_test  = get_x_tilde(evaluate_basis_functions(l[i], X_test, X_train))
            
        w_0 = np.random.randn(X_tilde_train.shape[ 1 ])
        objective_function = partial(compute_average_ll, X_train, y_train)
        w_map, hessian = find_map_and_hessian(objective_function, w_0)



if __name__ == "__main__":
    print("FTR")
    ftr()