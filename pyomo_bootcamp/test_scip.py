"""Test SCIP Solver"""
from pyomo.environ import *

def get_scip():
    opt = SolverFactory("scip")
    opt._version_timeout = 10
    return opt

model = ConcreteModel()
model.x = Var(initialize=1.0)
model.obj = Objective(expr=(model.x - 2)**2)

solver = get_scip()

results = solver.solve(model)

print(value(model.x))
