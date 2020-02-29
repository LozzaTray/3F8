import numpy as np


def logistic(x):
    """Logistic function""" 
    return 1.0 / (1.0 + np.exp(-x))


def predict(X_tilde, w):
    """
    Function that makes predictions with a logistic classifier

    Input:

    X_tile: matrix of input features (with a constant 1 appended to the left) 
            for which to make predictions
    w: vector of model parameters

    Output: The predictions of the logistic classifier
    """
    return logistic(np.dot(X_tilde, w))


def compute_average_ll(X_tilde, y, w):
    """
    Function that computes the average loglikelihood of the logistic classifier on some data.

    Input:

    X_tile: matrix of input features (with a constant 1 appended to the left) 
            for which to make predictions
    y: vector of binary output labels 
    w: vector of model parameters

    Output: The average loglikelihood
    """
    output_prob = predict(X_tilde, w)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))


def get_x_tilde(X): 
    """
    Function that expands a matrix of input features by adding a column equal to 1.

    Input:

    X: matrix of input features.

    Output: Matrix x_tilde with one additional constant column equal to 1 added.
    """
    return np.concatenate((np.ones((X.shape[ 0 ], 1 )), X), 1)


def compute_ll_grad(X_tilde, y, w):
    Xw = np.matmul(X_tilde, w)
    sigma_Xw = logistic(Xw)
    X_tilde_transpose = np.transpose(X_tilde)
    ll_grad = np.matmul(X_tilde_transpose, y - sigma_Xw)
    return ll_grad


def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    """
    Function that finds the model parameters by optimising the likelihood using gradient descent

    Input:

    X_tile_train: matrix of training input features (with a constant 1 appended to the left) 
    y_train: vector of training binary output labels 
    X_tile_test: matrix of test input features (with a constant 1 appended to the left) 
    y_test: vector of test binary output labels 
    alpha: step_size_parameter for the gradient based optimisation
    n_steps: the number of steps of gradient based optimisation

    Output: 

    1 - Vector of model parameters w 
    2 - Vector with average log-likelihood values obtained on the training set
    3 - Vector with average log-likelihood values obtained on the test set
    """
    w = np.random.randn(X_tilde_train.shape[ 1 ])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)

        ll_grad = compute_ll_grad(X_tilde_train, y_train, w)
        w = w + alpha * ll_grad# Gradient-based update rule for w. To be completed by the student

        ll_train[ i ] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[ i ] = compute_average_ll(X_tilde_test, y_test, w)
        #print(ll_train[ i ], ll_test[ i ])

    return w, ll_train, ll_test
