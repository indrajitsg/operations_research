import numpy as np
import pandas as pd
from pandas._config.config import options
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

# Estimate coeff of a simple linear regression using optimization

def get_data():
    """Generate data"""
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7],
                       'y': [0.5, 2.5, 2, 4, 3.5, 6, 5.5]})
    return df


def eval_fx(b0, b1, x):
    return b0 + b1 * x


def obj(beta, type='ss'):
    """Return model error (Average of Absolute Error"""
    vfunc = np.vectorize(eval_fx)
    df = get_data()
    y_hat = vfunc(beta[0], beta[1], df['x'])
    if type == 'abs':
        error = np.abs(y_hat - df['y'])
        return np.mean(error)
    else:
        error = np.sum((y_hat - df['y'])**2)
        return error

bounds = Bounds([-20.0, -20.0], [20.0, 20.0])

x0 = np.array([1.5, -1.5])

res = minimize(fun=obj,
               x0=x0,
               method='SLSQP',
               options={'verbose': 1, 'maxiter': 100},
               bounds=bounds)
print(res)

print(res['x'])

# Check with sklearn LinearRegression
lm1 = LinearRegression(fit_intercept=True, normalize=False)

df = get_data()
df.head()

lm1.fit(X=df[['x']], y=df['y'])
lm1.intercept_, lm1.coef_

