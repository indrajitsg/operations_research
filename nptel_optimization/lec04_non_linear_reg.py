import numpy as np
import pandas as pd
from pandas._config.config import options
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from scipy.optimize import broyden1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

# Estimate coeff of a multiple linear regression using optimization
def get_data():
    """Generate data"""
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4],
                       'y': [1.5, 2.5, 3.5, 5, 7.5]})
    return df


def eval_fx(C, A, x):
    return C * np.exp(A * x)


def obj(beta):
    """Return model error (Average of Absolute Error)"""
    vfunc = np.vectorize(eval_fx)
    df = get_data()
    y_hat = vfunc(beta[0], beta[1], df['x'])
    error = np.sum((y_hat - df['y'])**2)
    return error


bounds = Bounds([-5.0, -2.0], [5.0, 2.0])

x0 = np.array([1.5, 0.5])

# res = broyden1(F=obj, xin=x0, maxiter=200, verbose=True)

res = minimize(fun=obj,
               x0=x0,
               method='BFGS',
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
