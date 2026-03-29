"""Test Ipopt Installation"""
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(initialize=1.0)

model.obj = Objective(expr=(model.x - 2)**2)

SolverFactory("ipopt").solve(model)

print(value(model.x))
