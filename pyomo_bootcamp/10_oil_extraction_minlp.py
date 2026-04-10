"""Oil Extraction MINLP

Oilco must determine how many barrels of oil to extract during each of the next two years. If Oilco
extracts x1 million barrels during year 1, each barrel can be sold for (30 - x1) dollars. If Oilco
extracts x2 million barrels during year 2, each barrel can be sold for (35 - x2) dollars. The cost
of extracting x1 million barrels during year 1 is x1^2 million dollars, and the cost of extracting
x2 million barrels during year 2 is 2*x2^2 million dollars. 

A total of 20 million barrels of oil are available, and at most $250 million can be spent on
extraction. Formulate a MINLP to help Oilco maximize profits (revenues less costs) for the next two
years.
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

    # Sets
    model.year = pyo.RangeSet(2)
    year = model.year

    # Parameters
    model.max_barrels = pyo.Param(initialize=20)
    max_barrels = model.max_barrels
    
    model.max_budget = pyo.Param(initialize=250)
    max_budget = model.max_budget

    # Decision variables (millions of barrels)
    model.x = pyo.Var(year, within=pyo.NonNegativeIntegers)
    x = model.x

    # Objective function: Worker Cost + Setup Cost
    def objective_rule(model):
        revenue_y1 = x[1] * (30 - x[1])
        revenue_y2 = x[2] * (35 - x[2])
        cost_y1 = x[1]**2
        cost_y2 = 2*x[2]**2
        profit = revenue_y1 + revenue_y2 - cost_y1 - cost_y2
        return profit

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def max_oil_constraint_rule(model):
        return sum(x[i] for i in year) <= max_barrels
    
    model.MaxOilConstraint = pyo.Constraint(rule=max_oil_constraint_rule)

    def max_budget_constraint_rule(model):
        return x[1]**2 + 2*x[2]**2 <= max_budget

    model.MaxBudgetConstraint = pyo.Constraint(rule=max_budget_constraint_rule)
    
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
    for i in model.year:
        print(f"# of Barrels in Year {i}: {model.x[i]()}")

if __name__ == "__main__":
    main()

