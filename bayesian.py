from prob_utils import logistic
import numpy as np
from scipy import optimize

def kappa(sigma_squared):
    return (1 + np.pi * sigma_squared / 8) ** (-0.5)


def mu_a(x, w_map):
    wt_map = np.transpose(w_map)
    return np.matmul(wt_map, x)


def sigma_squared_a(x, Sn):
    xt = np.transpose(x)
    Sn_x = np.matmul(Sn, x)
    return np.matmul(xt, Sn_x)


def prediction(x, w_map, Sn):
    s_2 = sigma_squared_a(x, Sn)
    k = kappa(s_2)
    m = mu_a(x, w_map)
    return logistic(k * m)


def find_map_and_hessian(objective_function, w_0):
    inverted_fn = lambda x: (-1) * objective_function(x)
    w_map, objective_min, diction  = optimize.fmin_l_bfgs_b(inverted_fn, w_0)
    return w_map, None
