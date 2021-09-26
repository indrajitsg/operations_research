import numpy as np
import pandas as pd
from pandas._config.config import options
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

# Estimate coeff of a simple linear regression using optimization

def get_data():
    """Generate data"""
    df = pd.DataFrame({'x': [0, 2, 4, 5, 8, 9],
                       'y': [2, 3.2, 3.8, 4.6, 6.2, 6.8]})
    return df

def eval_fx(b0, b1, x):
    return b0 + b1 * x

def obj(beta):
    vfunc = np.vectorize(eval_fx)
    df = get_data()
    y_hat = vfunc(beta[0], beta[1], df['x'])
    # n = df.shape[0]
    error_abs = np.abs(y_hat - df['y'])
    return np.mean(error_abs)

bounds = Bounds([-10.0, -10.0], [10.0, 10.0])

x0 = np.array([1.5, -1.5])

res = minimize(fun=obj,
               x0=x0,
               method='SLSQP',
               options={'verbose': 1, 'maxiter': 100},
               bounds=bounds)
print(res)

print(res['x'])
