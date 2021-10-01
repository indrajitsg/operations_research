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
    df = pd.DataFrame({'x1': [0, 2, 2.5, 1, 4, 9, 8],
                       'x2': [0, 1, 2, 3, 6, 2, 4],
                       'y': [5, 10, 9, 0, 3, 27, 15]})
    return df


def eval_fx(b0, b1, b2, x1, x2):
    return b0 + b1 * x1 + b2 * x2


def obj(beta):
    """Return model error (Average of Absolute Error)"""
    vfunc = np.vectorize(eval_fx)
    df = get_data()
    y_hat = vfunc(beta[0], beta[1], beta[2], df['x1'], df['x2'])
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
df.head()

lm1.fit(X=df[['x1', 'x2']], y=df['y'])
lm1.intercept_, lm1.coef_

