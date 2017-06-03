# Performing imports
from src.RandomCoordinateDescent import *

import pandas as pd
from sklearn import preprocessing
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

hitters = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv', sep=',', header=0)

hitters = hitters.dropna()

X = hitters.drop('Salary', axis=1)
Y = hitters.Salary

X = pd.get_dummies(X, drop_first=True)  # drop_first=True says we only want k-1 dummies out of k categorical levels

# Centring and scaling the inputs. Only centering outputs
Xfull = preprocessing.scale(X)
Y = preprocessing.scale(Y, with_std=False)

# Initialising the model
l=1
b_init = np.zeros(Xfull.shape[1]).T
max_iterations = 1000

b, betas, rand_objs = randcoorddescent (Xfull, Y, b_init, l, max_iterations)

# Plotting objective function to ensure convergence
plt.plot(rand_objs[0:1000])
plt.title("Objective Function")
plt.show()

print("The coefficients for my function for lambda = 1 are :\n", b)

#We can see that we get very similar coefficients

#Doing CV to find best lambda
x_train, x_test, y_train, y_test = train_test_split(Xfull,Y, random_state=0)

lambda_vals = [10**k for k in range(-5, 5)]
test_errors_mylasso = np.zeros(10)
for i in range(0,10):
    li=lambda_vals[i]
    b,betas, rand_objs = randcoorddescent (x_train, y_train, b_init, li, max_iterations)
    test_errors_mylasso[i] = (y_test - np.dot(x_test,b)).T.dot(y_test - np.dot(x_test,b))

print("Test Errors from my algorithm:\n", test_errors_mylasso)

plt.scatter(np.arange(-5,5),test_errors_mylasso, label='My Errors',c='red')
plt.legend(loc='upper left')
plt.title('Test error for my function vs log(lambda)')
plt.show()