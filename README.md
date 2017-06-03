# Data558

Files included:

RandomCoordinateDescent.py : Performs Random Coordinate Descent to find coefficients for lasso linear regression.
It accepts the input X and Y along with an initial estimate of coefficients (advised to be zero), 
lambda for lasso and max_iterations.
The stoppage condition for the optimizing algorithm is number of iteration. Each iteration
 consists of optimizing all the parameters one after the other in a random manner.

simulation.py : Performs random coordinate descent on simulated data, 
shows how objective function converges for lambda = 1 (and prints estimated coefficients) 
and finally performs cross validation, showing the  validation error

real.py : Performs random coordinate descent on Hitters data from ISLR to predict salary of hitters, 
shows how objective function converges for lambda = 1 (and prints estimated coefficients) 
and finally performs cross validation, showing the  validation error. 
The real data is downloaded from https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv 

comparewithscikit.py : Compares the result of my algorithm with scikit on my simulated data by comparing the coefficients for lambda=1
 as well as the validation errors.

### Packages needed:

numpy

copy

pandas

sklearn

matplotlib

### How to run

Simply call the python file in python 3 to run it


