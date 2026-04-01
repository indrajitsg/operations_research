"""Leather Limited LP Problem

Leather Limited manufactures two type of belts: the deluxe model and the regular model. Each type
require 1 sq yard of leather. A regular belt requires 1 hour of skilled labor, and a deluxe belt
requires 2 hours. Each week, 40 sq yards of leather and 60 hours of skilled labor are available.
Each regular belt contributes to $3 of profit and each deluxe belt, $4. Formulate an LP to maximize
the profit.
"""
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def print_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_model():
    # Defining model
    model = pyo.ConcreteModel()

    # Decision variables
    model.x1 = pyo.Var(within=pyo.NonNegativeReals)
    model.x2 = pyo.Var(within=pyo.NonNegativeReals)

    x1 = model.x1
    x2 = model.x2

    # Objective function
    model.Obj = pyo.Objective(expr=4*x1 + 3*x2, sense=pyo.maximize)

    # Constraints
    model.cons1 = pyo.Constraint(expr=x1 + x2 <= 40)
    model.cons2 = pyo.Constraint(expr=2*x1 + x2 <= 60)
    return model


def choose_solver(solver_name):
    """Choose solver. Possible candidates:
        "gurobi", "cplex_direct", "knitroampl", "baron", "highs", "scip", "ipopt"
    """
    try:
        solver = SolverFactory(solver_name)
        if solver is not None and solver.available(False):
            return solver_name, solver
    except Exception:
        pass
    return None, None


def main():
    # Build model
    model = build_model()

    # Select solver
    if len(sys.argv) > 1:
        user_solver_name = sys.argv[1]
        print(f"Solver provided: {user_solver_name}")
    else:
        print("No solver was specified, defaulting to Highs")
        user_solver_name = "highs"
    solver_name, solver = choose_solver(user_solver_name)

    # Solve
    print_banner("STARTING SOLVE")
    print(f"Attempting to solve with solver: {solver_name}")
    results = solver.solve(model, tee=True)

    # Display solution
    print_banner("RESULTS")
    print(results)
    print(f"Objective: {model.Obj()}")
    print(f"x1 = {model.x1()}")
    print(f"x2 = {model.x2()}")


if __name__ == "__main__":
    main()

