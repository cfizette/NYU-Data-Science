import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tqdm
import math


### Assignment Owner: Tian Wang


#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test - test set, a 2D numpy array of size (num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # Remove columns with constant values
    cols_to_delete = np.all(train==train[0,:], axis=0)
    cols_to_delete = np.argwhere(cols_to_delete==True)
    train = np.delete(train, cols_to_delete, 1)
    test = np.delete(test, cols_to_delete, 1)

    min_arr = np.amin(train, axis=0)
    range_arr = np.amax(train, axis=0) - min_arr

    train_normalized = (train-min_arr)/range_arr
    test_normalized = (test-min_arr)/range_arr

    return train_normalized, test_normalized


#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    y_pred = np.matmul(X,theta)

    #return (1/len(y)) * np.matmul((y_pred - y).T, (y_pred - y))

    return np.mean(np.square(y - y_pred))


#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    y_pred = np.matmul(X, theta)
    n = len(y)
    return (2/n) * np.matmul(X.T, y_pred - y)


#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    
    hs = np.eye(num_features)
    
    for i, h in enumerate(hs):
        approx_grad[i] = (compute_square_loss(X, y, theta + epsilon*h) - compute_square_loss(X, y, theta - epsilon*h))/(2*epsilon)
        
    dist = np.linalg.norm(approx_grad - true_gradient)
    
    return dist <= tolerance


#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    true_gradient = gradient_func(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    
    hs = np.eye(num_features)
    
    for i, h in enumerate(hs):
        approx_grad[i] = (objective_func(X, y, theta + epsilon*h) - objective_func(X, y, theta - epsilon*h))/(2*epsilon)
        
    dist = np.linalg.norm(approx_grad - true_gradient)
    
    return dist <= tolerance


#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array, (num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    theta = np.zeros(num_features) #Initialize theta
    
    for i in range(num_step+1):
        loss_hist[i] = compute_square_loss(X, y, theta)
        theta_hist[i] = theta

        grad = compute_square_loss_gradient(X, y, theta)

        if grad_check:
            if not grad_checker(X, y, theta):
                warnings.warn('Error computing gradient on iteration {}'.format(i))
                return theta_hist, loss_hist
 
        theta -= grad*alpha

    return theta_hist, loss_hist


#######################################
### Backtracking line search
#Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
def check_ag_condition(X, y, theta, current_loss, alpha, c, p, grad_norm):
    # Checks Armijo–Goldstein condition for linear regression
    # Returns true if condition not satisfied
    return current_loss - compute_square_loss(X, y, theta-alpha*p) < c*alpha*grad_norm

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm , norm

def backtracking_line_search(X, y, max_alpha=1, b=0.5, c=0.5, num_step=1000):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    theta = np.zeros(num_features) #Initialize theta

    for i in range(num_step + 1):
        alpha = max_alpha
        loss_hist[i] = compute_square_loss(X, y, theta)
        theta_hist[i] = theta

        #precompute to avoid unnecessary computation
        current_loss = compute_square_loss(X, y, theta) 
        grad = compute_square_loss_gradient(X, y, theta)
        p, grad_norm = normalize(grad)

        # While Armijo-Goldstein condition is not satisfied, shrink alpha 
        while check_ag_condition(X, y, theta, current_loss, alpha, c, grad, grad_norm):
            alpha = b*alpha

        theta -= alpha*p

    return theta_hist, loss_hist


#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    square_loss_gradient = compute_square_loss_gradient(X, y, theta)
    regularization_term = 2 * lambda_reg * theta.T
    return square_loss_gradient + regularization_term


#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    
    for i in range(num_step+1):
        loss_hist[i] = compute_square_loss(X, y, theta)
        theta_hist[i] = theta

        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)

        theta -= grad*alpha

    return theta_hist, loss_hist


#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=10**-2, num_epoch=1000, C=0.1, averaged=False, eta_0=None):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch (num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size (num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist

    t=1
    mode=alpha

    for i in range(num_epoch):

        for j, (x_j, y_j) in enumerate(zip(X, y)):
            x_j = np.array([x_j])
            y_j = np.array([y_j])
            grad = compute_regularized_square_loss_gradient(x_j, y_j, theta, lambda_reg)

            # Adaptive step size methods
            if mode == '1/sqrt(t)':
                alpha = C/math.sqrt(t)
            if mode == '1/t':
                alpha = C/t
            if eta_0:
                alpha = eta_0/(1+eta_0*lambda_reg)

            theta -= grad*alpha # Update theta
            theta_hist[i,j] = theta
            loss_hist[i,j] = compute_square_loss(X, y, theta)

            t += 1

    if averaged:
        theta_hist = theta_hist.reshape((num_epoch*num_instances, num_features))
        return theta_hist.mean(axis=0), loss_hist
    return theta_hist, loss_hist
    


def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO

if __name__ == "__main__":
    main()
