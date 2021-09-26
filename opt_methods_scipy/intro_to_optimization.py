# Introduction to Optimization
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.special as ss
import mystic
import mystic.models as models

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


# -------------------------------------------------------------
#
# -------------------------------------------------------------

# -------------------------------------------------------------
#
# -------------------------------------------------------------

# -------------------------------------------------------------
#
# -------------------------------------------------------------
