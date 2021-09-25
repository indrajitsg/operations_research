import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

bounds = Bounds([5.0, 0.0], [9.0, 0.8])

def obj(x):
    return 9.82 * x[0] * x[1] + 2 * x[0]

linear_cons = LinearConstraint([[-1,  0],
                                [ 1,  0],
                                [ 0, -1],
                                [ 0,  1]],
                               [-np.inf, -np.inf, -np.inf, -np.inf],
                               [-2, 14, -0.2, 0.8])

def cons_f(x):
    c1 = 2500 / (3.141 * x[0] * x[1]) - 500
    c2 = 2500 / (3.141 * x[0] * x[1]) - 3.141**2 * (x[0]**2 + x[1]**2)/0.5882
    return [c1, c2]

non_linear_cons = NonlinearConstraint(cons_f, -np.inf, 0, hess=BFGS())

# Set initial values
x0 = np.array([7.0, 0.4])

# Solve
res = minimize(obj, x0, method='trust-constr', 
               constraints=[linear_cons, non_linear_cons],
               options={'verbose': 1}, bounds=bounds)

# Solutions
res['x']

res['constr']