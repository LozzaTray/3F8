from prob_utils import logistic, predict
import numpy as np
from scipy import optimize
from functools import partial


def log_prior(w, var_0):
    dim = len(w)
    return - (np.dot(w, w) / (2*var_0)) - (dim/2) * np.log(2*np.pi*var_0)


def log_likelihood(X_tilde, y, w):
    output_prob = predict(X_tilde, w)
    return np.sum(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))


def minus_log_f(X_tilde, y, var_0, w):
    return -(log_prior(w, var_0) + log_likelihood(X_tilde, y, w))


def calc_hessian(X_tilde, w, var_0):
    """
    Returns hessian of log f evaluated at given w (in this case w_map)
    """
    dim = X_tilde.shape[1]
    probs = predict(X_tilde, w)[:, np.newaxis] # turns into column matrix (vector)
    prob_var_array = np.matmul(probs, np.transpose(1-probs))
    diag = np.diag(np.diag(prob_var_array)) # extracts diagonal and keeps as matrix
    X_tilde_transpose = np.transpose(X_tilde)
    likelihood_term = np.matmul(X_tilde_transpose, np.matmul(diag, X_tilde))
    prior_term = np.identity(dim) * (1/var_0)
    return prior_term + likelihood_term 

def calc_Z(f_max, hessian_inverse):
    dim = hessian_inverse.shape[0]
    det_A_inv = np.linalg.det(hessian_inverse)
    log_Z =  np.log(f_max) + (dim/2)* np.log(2*np.pi) + (1/2)*np.log(det_A_inv)


def find_map(X_tilde, y, var_0):
    """
    Returns map solution and value of f evaluated at w_map
        return w_map, f_max, predictor_func
    """
    w_0 = np.random.randn(X_tilde.shape[ 1 ])
    minus_log_f_partial = partial(minus_log_f, X_tilde, y, var_0)
    w_map, min_minus_log_f, _diction = optimize.fmin_l_bfgs_b(minus_log_f_partial, w_0, approx_grad=True)
    f_max = np.exp(-min_minus_log_f)
    hessian = calc_hessian(X_tilde, w_map, var_0)
    hessian_inverse = np.linalg.inv(hessian)
    Z = 0 #calc_Z(f_max, hessian)
    predictor_function = partial(predictive_dist, hessian_inverse, w_map)
    return w_map, Z, predictor_function


def predictive_dist(A_inverse, w_map, X_tilde):
    """Predictive distribution for a new matrix X"""
    X_tilde_T = np.transpose(X_tilde)
    var_a = np.diag(np.matmul(X_tilde, np.matmul(A_inverse, X_tilde_T)))
    denominator = np.sqrt(1 + (np.pi / 8) * var_a)
    numerator = np.dot(X_tilde, w_map)
    return logistic(np.divide(numerator, denominator))


