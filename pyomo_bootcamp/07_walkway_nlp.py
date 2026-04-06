"""Walkway Area NLP

The intention is to place a handrail on a terrace with an empty space in its center in such a way
that the handrail goes around the interior and exterior terrace perimeters. This empty space, which
is rectangular in shape, measures:

    1. 10 m wide
    2. 18 m long

Between the interior and the exterior terrace perimeters, a walkway must be left, which is the same
width on all sides. If the handrail measures 250 m, formulate a non-linear programming model that
maximizes the area occupied by the walkway to improve circulation in it.
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

    # Parameters
    model.width = pyo.Param(initialize=10.0)
    model.length = pyo.Param(initialize=18.0)
    model.handrail = pyo.Param(initialize=250.0)

    width = model.width
    length = model.length
    handrail = model.handrail

    # Decision variables
    model.x = pyo.Var(within=pyo.NonNegativeReals)
    x = model.x

    # Objective function: Maximize the difference of the area of two rectangles
    def objective_rule(model):
        return (length + 2*x) * (width + 2*x) - (length * width)
        # return 2*(x * length) + 2*(x * width) + 4*x*x

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def handrail_constraint_rule(model):
        inner_perimeter = 2 * (length + width)
        outer_perimeter = 2 * ((length + 2*x) + (width + 2*x))
        return inner_perimeter + outer_perimeter <= handrail
    
    model.HandRailLimitConstraint = pyo.Constraint(rule=handrail_constraint_rule)
    
    return model


def choose_solver(solver_name):
    """Choose solver. Possible candidates:
        "gurobi", "cplex_direct", "knitroampl", "baron", "highs", "scip", "ipopt"
    """
    try:
        solver = SolverFactory(solver_name)
        if solver_name == "knitroampl":
            solver.options["par_numthreads"] = 2
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
    print(f"Width of Walkway: {model.x()}")


if __name__ == "__main__":
    main()
