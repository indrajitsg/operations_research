# main.py
"""
A simple Linear Programming (LP) example in Pyomo.

Problem:
    Maximize profit = 40*x + 30*y

Subject to:
    2*x + 1*y <= 40    (Labor)
    1*x + 1*y <= 30    (Material)
    1*x + 0*y <= 20    (Demand for x)
    x, y >= 0

Expected optimal solution:
    x = 10
    y = 20
    objective = 1000

How to run:
    python main.py

Notes:
- This script tries a few common LP solvers in order:
    HiGHS, GLPK, CBC, Gurobi, CPLEX
- Make sure you have:
    pip install pyomo
  and at least one solver installed and available on PATH.
"""

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    NonNegativeReals,
    maximize,
    SolverFactory,
    value,
    Suffix,
)
from pyomo.opt import TerminationCondition, SolverStatus


def build_model():
    """Build and return a simple LP model."""
    model = ConcreteModel(name="Simple_Product_Mix_LP")

    # ============================================================
    # PARAMETERS
    # ============================================================
    # Profit per unit
    profit_x = 40
    profit_y = 30

    # Resource limits
    labor_limit = 40
    material_limit = 30
    demand_x_limit = 20

    # ============================================================
    # DECISION VARIABLES
    # ============================================================
    # x = units of product X to produce
    # y = units of product Y to produce
    model.x = Var(domain=NonNegativeReals)
    model.y = Var(domain=NonNegativeReals)

    # ============================================================
    # CONSTRAINTS
    # ============================================================
    # Labor constraint
    model.labor = Constraint(expr=2 * model.x + 1 * model.y <= labor_limit)

    # Material constraint
    model.material = Constraint(expr=1 * model.x + 1 * model.y <= material_limit)

    # Demand limit for product X
    model.demand_x = Constraint(expr=1 * model.x <= demand_x_limit)

    # ============================================================
    # OBJECTIVE
    # ============================================================
    # Maximize total profit
    model.obj = Objective(
        expr=profit_x * model.x + profit_y * model.y,
        sense=maximize,
    )

    # Optional suffixes: these may be populated by some solvers for LPs
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)  # reduced costs (if solver provides)

    return model


def choose_solver():
    """
    Try a few common solvers and return the first available one.
    """
    candidate_solvers = ["knitroampl", "baron", "highs", "scip", "ipopt", "gurobi"]

    for solver_name in candidate_solvers:
        try:
            solver = SolverFactory(solver_name)
            if solver is not None and solver.available(False):
                return solver_name, solver
        except Exception:
            pass

    return None, None


def print_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_model_summary(model):
    print_banner("MODEL SUMMARY")
    print(f"Model name              : {model.name}")
    print(f"Number of variables     : {len(list(model.component_data_objects(Var, active=True)))}")
    print(f"Number of constraints   : {len(list(model.component_data_objects(Constraint, active=True)))}")
    print("Problem type            : Linear Program (LP)")
    print("Objective sense         : Maximize")


def print_solver_summary(results, solver_name):
    print_banner("SOLVER SUMMARY")
    print(f"Solver used             : {solver_name}")

    solver_info = results.solver
    print(f"Status                  : {solver_info.status}")
    print(f"Termination condition   : {solver_info.termination_condition}")

    # These fields may or may not exist depending on solver
    if hasattr(solver_info, "time"):
        print(f"Solve time              : {solver_info.time}")

    if hasattr(solver_info, "return_code"):
        print(f"Return code             : {solver_info.return_code}")


def print_solution(model):
    print_banner("OPTIMAL SOLUTION")
    print(f"Objective value (profit): {value(model.obj):.4f}")
    print(f"x (Product X)           : {value(model.x):.4f}")
    print(f"y (Product Y)           : {value(model.y):.4f}")


def print_constraint_details(model):
    print_banner("CONSTRAINT DETAILS")

    for con in model.component_data_objects(Constraint, active=True):
        body_val = value(con.body)

        ub = con.upper
        lb = con.lower

        # Compute slack information
        upper_slack = None if ub is None else value(ub) - body_val
        lower_slack = None if lb is None else body_val - value(lb)

        print(f"Constraint              : {con.name}")
        print(f"  Body value            : {body_val:.4f}")

        if lb is not None:
            print(f"  Lower bound           : {value(lb):.4f}")
            print(f"  Lower slack           : {lower_slack:.4f}")

        if ub is not None:
            print(f"  Upper bound           : {value(ub):.4f}")
            print(f"  Upper slack           : {upper_slack:.4f}")

        # Dual value, if available from solver
        dual_val = model.dual.get(con, None)
        if dual_val is not None:
            print(f"  Dual / Shadow price   : {dual_val:.4f}")
        else:
            print(f"  Dual / Shadow price   : Not available from this solver")

        print("-" * 70)


def print_variable_details(model):
    print_banner("VARIABLE DETAILS")

    for var in model.component_data_objects(Var, active=True):
        var_val = value(var)
        lb = var.lb
        ub = var.ub

        print(f"Variable                : {var.name}")
        print(f"  Value                 : {var_val:.4f}")
        print(f"  Lower bound           : {lb if lb is not None else '-inf'}")
        print(f"  Upper bound           : {ub if ub is not None else '+inf'}")

        rc_val = model.rc.get(var, None)
        if rc_val is not None:
            print(f"  Reduced cost          : {rc_val:.4f}")
        else:
            print(f"  Reduced cost          : Not available from this solver")

        print("-" * 70)


def main():
    print_banner("PYOMO SIMPLE LP EXAMPLE")

    # Build model
    model = build_model()
    print_model_summary(model)

    # Choose solver
    solver_name, solver = choose_solver()
    if solver is None:
        print_banner("ERROR")
        print("No supported solver was found.")
        print("Please install at least one LP solver and make sure it is on PATH.")
        print("\nCommon options:")
        print("  - HiGHS")
        print("  - GLPK")
        print("  - CBC")
        print("  - Gurobi")
        print("  - CPLEX")
        return

    # Solve
    print_banner("STARTING SOLVE")
    print(f"Attempting to solve with solver: {solver_name}")

    # tee=True streams the solver log to the terminal, similar to commercial solvers
    results = solver.solve(model, tee=True)

    # Print summary
    print_solver_summary(results, solver_name)

    # Check solve status
    is_ok = (
        results.solver.status == SolverStatus.ok
        and results.solver.termination_condition == TerminationCondition.optimal
    )

    if not is_ok:
        print_banner("NO OPTIMAL SOLUTION FOUND")
        print("The solver did not report an optimal solution.")
        return

    # Print solution details
    print_solution(model)
    print_constraint_details(model)
    print_variable_details(model)

    print_banner("DONE")
    print("LP solved successfully.")


if __name__ == "__main__":
    main()