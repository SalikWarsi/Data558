# Performing imports
from src.RandomCoordinateDescent import *
from sklearn import preprocessing
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Generate Random data
np.random.seed(0)
x = np.random.normal(size=(100))
err = np.random.normal(size=(100))

# let b0 = 1, b1=3, b2=4, b3=2
Y = 1 + 3*x + 4*(x**2) + 2*(x**3) + err

Xfull = np.zeros((100,10))
Xfull[:,0] = x
Xfull[:,1] = x**2
Xfull[:,2] = x**3
Xfull[:,3] = x**4
Xfull[:,4] = x**5
Xfull[:,5] = x**6
Xfull[:,6] = x**7
Xfull[:,7] = x**8
Xfull[:,8] = x**9
Xfull[:,9] = x**10

# Centring and scaling the inputs
Xfull = preprocessing.scale(Xfull)
Y = preprocessing.scale(Y)

# Initialising the model
l=1
b_init = np.zeros(Xfull.shape[1]).T
max_iterations = 1000

b, betas, rand_objs = randcoorddescent (Xfull, Y, b_init, l, max_iterations)

print("The coefficients from my algorithm for lambda =1 are :\n", b)

#Doing Lasso from SciKit
lasso_model = Lasso(alpha=l/2, fit_intercept=False)
lasso_model.fit(Xfull,Y)
print("The coefficients from scikit for the equivalent lambda are :\n", lasso_model.coef_)

#We can see that we get very similar coefficients

#Doing CV to find best lambda
x_train, x_test, y_train, y_test = train_test_split(Xfull,Y, random_state=0)

lambda_vals = [10**k for k in range(-5, 5)]
test_errors_mylasso = np.zeros(10)
test_errors_scikitlasso = np.zeros(np.shape(lambda_vals))
for i in range(0,10):
    li=lambda_vals[i]
    b,betas, rand_objs = randcoorddescent (x_train, y_train, b_init, li, max_iterations)
    test_errors_mylasso[i] = (y_test - np.dot(x_test,b)).T.dot(y_test - np.dot(x_test,b))
    lasso_model = Lasso(alpha=li / 2, fit_intercept=False)
    lasso_model.fit(x_train, y_train)
    b_scikit = lasso_model.coef_
    test_errors_scikitlasso[i] = (y_test - np.dot(x_test, b_scikit)).T.dot(y_test - np.dot(x_test, b_scikit))

print("Test Errors from my algorithm:\n", test_errors_mylasso)
print("Test Errors from scikit:\n", test_errors_scikitlasso)

plt.scatter(np.arange(-5,5),test_errors_mylasso, label='My Errors',c='red')
plt.scatter(np.arange(-5,5),test_errors_scikitlasso, label='SciKit Errors',c='blue')
plt.legend(loc='upper left')
plt.title('Test error vs Log of lambda for My errors and Scikit errors')
plt.show()