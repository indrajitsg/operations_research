"""Test Non-Convex Optimization"""
import sys
#from math import sin, sqrt
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
    model.x = pyo.Var(bounds=(-10, 10), initialize=-10.0)
    model.y = pyo.Var(bounds=(-10, 10), initialize=-10.0)

    # Objective function
    def objective_rule(model):
        return model.x * pyo.sin(pyo.sqrt(abs(model.x))) + model.y * pyo.sin(pyo.sqrt(abs(model.y)))

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraint
    model.con = pyo.Constraint(expr=model.x**2 + model.y**2 <= 100)
    
    return model


def choose_solver(solver_name):
    """Choose solver. Possible candidates:
        "gurobi", "cplex_direct", "knitroampl", "baron", "highs", "scip", "ipopt"
    """
    try:
        solver = SolverFactory(solver_name)
        if solver_name == "knitroampl":
            solver.options["par_numthreads"] = 2
            solver.options['ms_enable'] = 1
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
    print(f"x: {model.x()}")
    print(f"y: {model.y()}")

if __name__ == "__main__":
    main()
