from hw1_code import *
import pandas as pd 
import numpy as np
from numpy.testing import assert_array_almost_equal



def test_compute_square_loss():
    X = np.array([[1,2], [3,4]])
    theta = np.array([2,3])
    y = np.array([1,2])
    
    true_loss = 152.5
    calc_loss = compute_square_loss(X, y, theta)
    
    assert calc_loss == true_loss
    
def test_compute_square_loss_gradient():
    X = np.array([[1,2], [3,4]])
    theta = np.array([2,3])
    y = np.array([1,2])
    
    assert generic_gradient_checker( X, y, theta, compute_square_loss, compute_square_loss_gradient) == True
    assert grad_checker(X, y, theta) == True

def test_gradient_descent():
    df = pd.read_csv('data.csv')
    X = df.drop('y', axis=1)
    y = df.y

    theta_hist, _ = regularized_grad_descent(X, y, alpha=0.01, num_step=2000)

