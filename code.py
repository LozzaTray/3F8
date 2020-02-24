# This is the auxiliary code for the 3F8 coursework. Some parts are missing and
# should be completed by the student. These are Marked with XXX

import numpy as np
import matplotlib.pyplot as plt

# Load data
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')

# We randomly permute the data
permutation = np.random.permutation(X.shape[ 0 ])
X = X[ permutation, : ]
y = y[ permutation ]

# split into training and test sets
n_train = 800
X_train = X[ 0 : n_train, : ]
X_test = X[ n_train :, : ]
y_train = y[ 0 : n_train ]
y_test = y[ n_train : ]

def plot_data_internal(X, y):
    """
    Function that plots the points in 2D together with their labels

    Inputs:

    X: 2d array with the input features
    y: 1d array with the class labels (0 or 1)

    Output: 2D matrices with the x and y coordinates of the points shown in the plot
    """

    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy


def plot_data(X, y):
    """
    Function that plots the data without returning anything by calling "plot_data_internal".

    Input:

    X: 2d array with the input features
    y: 1d array with the class labels (0 or 1)

    Output: Nothing.
    """
    xx, yy = plot_data_internal(X, y)
    plt.show()


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


def plot_ll(ll_training, ll_test, title=''):
    """
    Function that plots the average log-likelihood returned by "fit_w"

    Input:

    ll: vector with log-likelihood values

    Output: Nothing
    """
    if title == '':
        title = 'Plot of Average Log-Likelihood Curve'
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll_training) + 2)
    plt.ylim(min(ll_training) - 0.1, max(ll_training) + 0.1)
    ax.plot(np.arange(1, len(ll_training) + 1), ll_training, 'r-', label="Training data")
    ax.plot(np.arange(1, len(ll_test) + 1), ll_test, 'b-', label="Test data")
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title(title)
    plt.legend()
    plt.show()


def compute_confusion_array(X_tilde, w, y):
    """computes confusion array"""
    output_probs = predict(X_tilde, w)
    assignments = (output_probs > 0.5)
    values = np.add(assignments, 2*y)

    types = [0, 0, 0, 0] # true negative = 0 , false positive = 1, false negative = 2, true positive = 3
    for i in range(0,4):
        types[i] = (values == i).sum()

    print("confusion array")
    print("| tn | fp|\n| fn | tp |")
    print("| {} | {}|\n| {} | {} |".format(types[0], types[1], types[2], types[3]))
    return types


def plot_predictive_distribution(X, y, w, map_inputs = lambda x : x, title=""):
    """
    Function that plots the predictive probabilities of the logistic classifier

    Input:

    X: 2d array with the input features for the data (without adding a constant column with ones at the beginning)
    y: 1d array with the class labels (0 or 1) for the data
    w: parameter vector
    map_inputs: function that expands the original 2D inputs using basis functions.

    Output: Nothing.
    """
    if title == "":
        title = "Predicitive Distribution"

    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict(X_tilde, w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.title(title)
    plt.show()


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


def linear_classifier():
    """Performs tasks on the linear classifier"""

    alpha = 0.001 # Learning rate
    n_steps = 100 # Number of steps

    X_tilde = get_x_tilde(X)
    X_tilde_train = get_x_tilde(X_train)
    X_tilde_test = get_x_tilde(X_test)

    print("Training linear classifier...")
    w, ll_train, ll_test = fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha)

    print("Plotting ll curve...")
    plot_ll(ll_train, ll_test, "Log-likelihood curve with linear classifier (alpha=" + str(alpha) + ")")

    print("ll_train | ll_test")
    print(str(ll_train[-1]) + " | " + str(ll_test[-1]))

    print("Computing linear classifier confusion on test data...")
    error_types = compute_confusion_array(X_tilde_test, w, y_test)

    print("Plotting predictive distribution...")
    plot_predictive_distribution(X, y, w)


def expanded_classifier(l, a, n):

    print("Expanding data with basis functions")    
    l = l # Width of the Gaussian basis funcction
    X_tilde = get_x_tilde(evaluate_basis_functions(l, X, X_train))
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
    error_types = compute_confusion_array(X_tilde_test, w, y_test)

    print("Plotting predicitive distribution...")
    plot_predictive_distribution(X, y, w, lambda x : evaluate_basis_functions(l, x, X_train))


print("Plotting data basic...")
plot_data(X, y)

print("Linear classifier:")
linear_classifier()

print("Expanded classifier:")
expanded_classifier(l=0.01, a=0.01, n=1000)
expanded_classifier(l=0.1, a=0.01, n=1000)
expanded_classifier(l=1, a=0.0001, n=1000)