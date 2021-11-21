import numpy as np
import pandas as pd
from pandas._config.config import options
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

# Estimate coeff of a multiple linear regression using optimization
def get_data():
    """Generate data"""
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 5],
                       'y': [2.1, 7.7, 13.6, 27.2, 40.9, 61.1]})
    return df


def eval_fx(b0, b1, b2, x):
    return b0 + b1 * x + b2 * x * x


def obj(beta):
    """Return model error (Average of Absolute Error)"""
    vfunc = np.vectorize(eval_fx)
    df = get_data()
    y_hat = vfunc(beta[0], beta[1], beta[2], df['x'])
    error = np.sum((y_hat - df['y'])**2)
    return error


bounds = Bounds([-20.0, -20.0, -20.0], [20.0, 20.0, 20.0])

x0 = np.array([1.5, -1.5, 0.7])

res = minimize(fun=obj,
               x0=x0,
               method='SLSQP',
               options={'maxiter': 200},
               bounds=bounds)
print(res)
print(res['x'])

# Check with sklearn LinearRegression
lm1 = LinearRegression(fit_intercept=True, normalize=False)

df = get_data()
df['x2'] = df['x'] * df['x']
df.head()

lm1.fit(X=df[['x', 'x2']], y=df['y'])
lm1.intercept_, lm1.coef_
