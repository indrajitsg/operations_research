# Introduction to Optimization
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as ss
import mystic
import mystic.models as models
import cvxopt as cvx
from cvxopt import solvers as cvx_solvers

# -------------------------------------------------------------
# Basic components
# -------------------------------------------------------------
objective = np.poly1d([1.3, 4.0, 0.6])

print(objective)

# Minimize a function using downhill simplex algorithm (Nelder-Mead Simplex Algo)
x_ = opt.fmin(func=objective, x0=[3])
print(f"solved: x={x_}")

# Plot the objective
x = np.linspace(-4, 1, 101)
plt.plot(x, objective(x))
plt.plot(x_, objective(x_), 'ro')
plt.show()

# -------------------------------------------------------------
# Optimize with box constraints
# -------------------------------------------------------------
x = np.linspace(2, 7, 200)

# 1st order Bessel
j1x = ss.j1(x)
plt.plot(x, j1x)

# Use scipy.optimize's more modern "results object" interface
result = opt.minimize_scalar(ss.j1, method='bounded', bounds=[2, 4])

j1_min = ss.j1(result.x)
plt.plot(result.x, j1_min, 'ro')
plt.show()

# -------------------------------------------------------------
# The Gradient and / or Hessian
# -------------------------------------------------------------
print(models.rosen.__doc__)
mystic.model_plotter(models.rosen, kwds='-f -d -x 1 -b "-3:3:.1, -1:5:.1, 1"')

# Optimize the Rosenbrock function
# initial guess
x0 = [1.3, 1.6, -0.5, -1.8, 0.8]

result = opt.minimize(opt.rosen, x0)
print(result['x'])

# number of function evaluations
print(result.nfev)

# optimize again, but this time provide the derivative
result = opt.minimize(opt.rosen, x0, jac=opt.rosen_der)
print(result['x'])

# number of function evaluations and derivative evaluations
print(result.nfev, result.njev)

# however, note for a different x0
for i in range(10):
    x0 = np.random.randint(-20, 20, 5)
    result = opt.minimize(opt.rosen, x0, jac=opt.rosen_der)
    print(f"{result.x} : {result.nfev}")

# -------------------------------------------------------------
# Using penalty function
# -------------------------------------------------------------
'''
Maximize f(x) = 2 * x0 * x1 + 2 * x0 - x0**2 - 2*x1**2
subject to: 
    x0**3 - x1 == 0
    x1         >= 1
'''

def objective(x, sign=1.0):
    return sign * (2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

def derivative(x, sign=1.0):
    dfdx0 = sign * (-2 * x[0] + 2 * x[1] + 2)
    dfdx1 = sign * (2 * x[0] - 4 * x[1])
    return np.array([dfdx0, dfdx1])

# Unconstrained
result = opt.minimize(fun=objective, x0=[-1.0, 1.0], args=(-1.0, ),
                      jac=derivative, method='SLSQP', options={'disp': True})
print(f"Unconstrained: {result.x}")
print(f"Unconstrained Objective: {result.fun}")

# Constrained
cons = ({'type': 'eq',
         'fun': lambda x: np.array([x[0]**3 - x[1]]),
         'jac': lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - 1]),
         'jac' : lambda x: np.array([0.0, 1.0])})


result = opt.minimize(objective, [-1.0,1.0], args=(-1.0,), jac=derivative,
                      constraints=cons, method='BFGS', options={'disp': True})

print("Constrained: {}".format(result.x))
print(f"Constrained Objective: {result.fun}")

# -------------------------------------------------------------
# Convex optimization
# -------------------------------------------------------------

# http://cvxopt.org/examples/tutorial/lp.html
'''
minimize:  f = 2*x0 + x1

subject to:
           -x0 + x1 <= 1
            x0 + x1 >= 2
            x1 >= 0
            x0 - 2*x1 <= 4

Ax <= b
A = A' (in NumPy)
'''

A = cvx.matrix([[-1.0, -1.0,  0.0,  1.0],
                [ 1.0, -1.0, -1.0, -2.0]])

b = cvx.matrix([ 1.0, -2.0, 0.0, 4.0 ])
cost = cvx.matrix([ 2.0, 1.0 ])
sol = cvx_solvers.lp(cost, A, b)

print(sol['x'])

m1 = np.array([[-1, 1], [-1, -1], [0, -1], [1, -2]])

# -------------------------------------------------------------
# Quadratic Programming
# -------------------------------------------------------------

# http://cvxopt.org/examples/tutorial/qp.html
'''
minimize:  f = 2*x1**2 + x2**2 + x1*x2 + x1 + x2

subject to:
            x1 >= 0
            x2 >= 0
            x1 + x2 == 1
'''

Q = 2*cvx.matrix([ [2, .5], [.5, 1] ])
p = cvx.matrix([1.0, 1.0])
G = cvx.matrix([[-1.0,0.0],[0.0,-1.0]])
h = cvx.matrix([0.0,0.0])
A = cvx.matrix([1.0, 1.0], (1,2))
b = cvx.matrix(1.0)
sol = cvx_solvers.qp(Q, p, G, h, A, b)

print(sol['x'])

# -------------------------------------------------------------
# Exercise
# -------------------------------------------------------------

'''
Minimize: f = -1x[0] + 4x[1]

Subject to:
        -3*x[0] +   x[1] <= 6
           x[0] + 2*x[1] <= 4
                    x[1] >= -3

where: -inf <= x[0] <= inf
'''

m1 = np.array([[-3.0, 1.0],
               [ 1.0, 2.0],
               [0.0, -1.0]])

A = cvx.matrix(m1)
b = cvx.matrix([6.0, 4.0, 3.0])
cost = cvx.matrix([-1.0, 4.0])
sol = cvx_solvers.lp(cost, A, b)

print(sol['x'])

# -------------------------------------------------------------
#
# -------------------------------------------------------------

# -------------------------------------------------------------
#
# -------------------------------------------------------------

# -------------------------------------------------------------
#
# -------------------------------------------------------------

# -------------------------------------------------------------
#
# -------------------------------------------------------------

# -------------------------------------------------------------
#
# -------------------------------------------------------------
