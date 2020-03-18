import numpy as np


def logistic_overflow_safe(x):
    """Logistic function. If function overflows, returns 0 for that value"""
    f = np.array([])
    with np.errstate(all='raise'):
        for i in x:
            try:
                val = 1.0 / (1.0 + np.exp(-i))
            except FloatingPointError:
                val = 0
            f = np.append(f, val)
    return f


def logistic(x):
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
        ll_grad = compute_ll_grad(X_tilde_train, y_train, w)
        w = w + alpha * ll_grad

        ll_train[ i ] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[ i ] = compute_average_ll(X_tilde_test, y_test, w)

    return w, ll_train, ll_test


def compute_confusion_array(X_tilde, w, y):
    """computes confusion array"""
    output_probs = predict(X_tilde, w)
    assignments = (output_probs > 0.5)
    values = np.add(assignments, 2*y)

    types = [0, 0, 0, 0] # true negative = 0 , false positive = 1, false negative = 2, true positive = 3
    for i in range(0,4):
        types[i] = (values == i).sum()

    return types


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


def get_confusion_matrix(y_hat, y):
    assignments = (y_hat > 0.5)
    values = np.add(assignments, 2*y)

    conf = [0, 0, 0, 0] # true negative = 0 , false positive = 1, false negative = 2, true positive = 3
    for i in range(0,4):
        conf[i] = (values == i).sum()

    true_zeroes = conf[0] + conf[1]
    true_ones = conf[2] + conf[3]

    matrix = [ 
        [conf[0] / true_zeroes , conf[1] / true_zeroes] ,
        [conf[2] / true_ones , conf[3] / true_ones]
    ]

    return matrix


def get_average_ll(y_hat, y):
    return np.mean(y * np.log(y_hat) + (1 - y) * np.log(1.0 - y_hat))