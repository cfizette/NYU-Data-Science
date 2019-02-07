from hw1_code import *



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
