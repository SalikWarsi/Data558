"""
Developed by salikw@uw.edu

This code performs the random coordinate descent algorithm to perform linear regression with lasso regularization

It accepts the input X and Y along with an initial estimate of coefficients (advised to be zero), 
lambda for lasso and max_iterations. The stoppage condition for the optimizing algorithm is number of iteration.

Each iteration consists of optimizing all the parameters one after the other in a random manner.
"""

import numpy as np
import copy


def computeobj(x, y, b, l):
    """
    :param x: the input parameters. Matrix of size n * d
    :param y: the training example output. Matrix of size n * 1
    :param b: the coefficients of size d*1
    :param l: scalar value for coefficient of regularization (lambda)
    :return: the value of objective function for linear regression with lasso regularization
    """
    n = y.shape[0]
    return ((1/n)*sum((y-x.dot(b))**2)) + l*sum(np.abs(b))


def pickcoord(d) :
    """
    Picks a random integer between 1 and d
    :param d: range from which to pick 
    :return: random integer
    """
    return np.random.randint(d) + 1


def randcoorddescent (x, y, b_init, l, max_iterations):
    """
    Performs Random Coordinate descent for lasso linear regression
    :param x: the input parameters. Matrix of size n * d
    :param y: the training example output. Matrix of size n * 1
    :param b_init: the initial coefficient estimate of size d*1
    :param l: scalar value for coefficient of regularization (lambda)
    :param max_iterations: max imum number of iterations
    :return: the final coefficients, the progress of coefficients at end of each iteration, the value of objective function at end of each iteration
    """
    n = y.shape[0]
    d = b_init.shape[0]
    b = copy.deepcopy(b_init)
    a = 2*np.sum(x ** 2, axis=0)/n
    rand_objs = np.zeros(max_iterations+1)
    rand_objs[0] = computeobj(x, y, b, l)
    betas = np.zeros((max_iterations+1, d))
    np.random.seed(0)
    for i in range(1, max_iterations+1):
        for jc in range(0, d):
            j = pickcoord(d)-1
            b[j] = 0  # This should remove the Beta j term from the sum
            xjs = x[:,j]
            cj = (xjs.T).dot(y-(x.dot(b)))*2/n
            if cj>l :
                b[j] = (cj-l)/a[j]
            elif cj < (-1*l) :
                b[j] = (cj+l)/a[j]
            else:
                b[j] = 0
        betas[i,:] = b
        rand_objs[i] = computeobj(x, y, b, l)
    return b, betas, rand_objs