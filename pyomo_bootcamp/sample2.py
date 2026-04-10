# main.py
"""
A simple LP example in Pyomo with Gurobi-like reporting.

Problem
-------
Maximize:
    40*x + 30*y

Subject to:
    2*x + 1*y <= 40   (labor)
    1*x + 1*y <= 30   (material)
    1*x         <= 20 (demand_x)
    x, y >= 0

Expected optimal solution:
    x = 10
    y = 20
    objective = 1000

Run:
    python main.py

Requirements:
    pip install pyomo

Also install at least one solver and make sure it is on PATH.
This script tries:
    1. appsi_highs
    2. highs
    3. glpk
    4. cbc
    5. gurobi
    6. cplex
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
from pyomo.opt import SolverStatus, TerminationCondition


# -------------------------------------------------------------------
# Utility formatting helpers
# -------------------------------------------------------------------
def line(char="=", width=78):
    return char * width


def section(title, char="="):
    print()
    print(line(char))
    print(title)
    print(line(char))


def sub_section(title):
    print()
    print(title)
    print(line("-"))


def fmt_num(x, digits=4):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_bound(x):
    if x is None:
        return "+inf/-inf"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def safe_value(expr):
    try:
        return value(expr)
    except Exception:
        return None


# -------------------------------------------------------------------
# Build the LP model
# -------------------------------------------------------------------
def build_model():
    model = ConcreteModel(name="Simple_Product_Mix_LP")

    # ===============================================================
    # PARAMETERS
    # ===============================================================
    # Profit coefficients
    profit_x = 40
    profit_y = 30

    # Resource / capacity limits
    labor_limit = 40
    material_limit = 30
    demand_x_limit = 20

    # ===============================================================
    # DECISION VARIABLES
    # ===============================================================
    # x = quantity of product X
    # y = quantity of product Y
    model.x = Var(domain=NonNegativeReals)
    model.y = Var(domain=NonNegativeReals)

    # ===============================================================
    # CONSTRAINTS
    # ===============================================================
    # Labor capacity
    model.labor = Constraint(expr=2 * model.x + 1 * model.y <= labor_limit)

    # Material capacity
    model.material = Constraint(expr=1 * model.x + 1 * model.y <= material_limit)

    # Demand bound for product X
    model.demand_x = Constraint(expr=1 * model.x <= demand_x_limit)

    # ===============================================================
    # OBJECTIVE
    # ===============================================================
    # Maximize total profit
    model.obj = Objective(expr=profit_x * model.x + profit_y * model.y, sense=maximize)

    # Optional suffixes to import duals / reduced costs if solver provides them
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)

    return model


# -------------------------------------------------------------------
# Solver selection
# -------------------------------------------------------------------
def choose_solver():
    candidates = ["knitroampl", "highs", "ipopt", "scip", "gurobi"]
    for name in candidates:
        try:
            solver = SolverFactory(name)
            if solver is not None and solver.available(False):
                return name, solver
        except Exception:
            pass
    return None, None


# -------------------------------------------------------------------
# Model statistics
# -------------------------------------------------------------------
def count_nonzeros_in_constraint(con):
    """
    For this small teaching example, we estimate the number of nonzeros
    from the linear expression string. This is not a symbolic exact count,
    but it works well enough for a simple LP reporting view.
    """
    expr_str = str(con.body)
    # crude count: count occurrences of variable names x/y in expression
    # good enough for this tiny demonstrator
    count = 0
    for token in ["x", "y"]:
        count += expr_str.count(token)
    return count


def print_model_statistics(model):
    vars_list = list(model.component_data_objects(Var, active=True))
    cons_list = list(model.component_data_objects(Constraint, active=True))

    n_vars = len(vars_list)
    n_cons = len(cons_list)

    n_continuous = 0
    n_binary = 0
    n_integer = 0

    for v in vars_list:
        if v.is_binary():
            n_binary += 1
        elif v.is_integer():
            n_integer += 1
        else:
            n_continuous += 1

    n_nonzeros = sum(count_nonzeros_in_constraint(c) for c in cons_list)

    sub_section("Model Statistics")
    print(f"{'Model name':30s}: {model.name}")
    print(f"{'Rows (constraints)':30s}: {n_cons}")
    print(f"{'Columns (variables)':30s}: {n_vars}")
    print(f"{'Nonzeros (estimated)':30s}: {n_nonzeros}")
    print(f"{'Continuous variables':30s}: {n_continuous}")
    print(f"{'Binary variables':30s}: {n_binary}")
    print(f"{'Integer variables':30s}: {n_integer}")
    print(f"{'Objective sense':30s}: Maximization")


def print_problem_definition():
    sub_section("Problem Definition")
    print("Maximize")
    print("    40*x + 30*y")
    print("Subject to")
    print("    labor    : 2*x + 1*y <= 40")
    print("    material : 1*x + 1*y <= 30")
    print("    demand_x : 1*x       <= 20")
    print("    x, y >= 0")


# -------------------------------------------------------------------
# Solve and reporting
# -------------------------------------------------------------------
def print_solver_header(solver_name):
    sub_section("Optimization Start")
    print("Reading model...")
    print("Checking available solver...")
    print(f"Selected solver               : {solver_name}")
    print("Building problem instance...")
    print("Invoking solver...")


def print_solver_summary(results, solver_name):
    sub_section("Solver Summary")

    print(f"{'Solver used':30s}: {solver_name}")
    print(f"{'Solver status':30s}: {results.solver.status}")
    print(f"{'Termination condition':30s}: {results.solver.termination_condition}")

    if hasattr(results.solver, "time"):
        print(f"{'Solve time':30s}: {results.solver.time}")
    if hasattr(results.solver, "return_code"):
        print(f"{'Return code':30s}: {results.solver.return_code}")
    if hasattr(results.solver, "message") and results.solver.message:
        print(f"{'Solver message':30s}: {results.solver.message}")


def is_optimal(results):
    return (
        results.solver.status == SolverStatus.ok
        and results.solver.termination_condition == TerminationCondition.optimal
    )


def print_solution_summary(model, results):
    obj_val = safe_value(model.obj)

    sub_section("Solution Summary")
    print(f"{'Solution status':30s}: OPTIMAL")
    print(f"{'Objective value':30s}: {fmt_num(obj_val)}")

    # For LPs, best bound = objective at optimality
    print(f"{'Best bound':30s}: {fmt_num(obj_val)}")
    print(f"{'Optimality gap (%)':30s}: 0.0000")


def print_variable_report(model, tol=1e-8):
    sub_section("Variable Report")
    header = (
        f"{'Variable':15s}"
        f"{'Value':>14s}"
        f"{'Lower Bound':>16s}"
        f"{'Upper Bound':>16s}"
        f"{'Reduced Cost':>16s}"
        f"{'At Bound?':>14s}"
    )
    print(header)
    print(line("-"))

    for v in model.component_data_objects(Var, active=True):
        val = safe_value(v)
        lb = v.lb
        ub = v.ub
        rc = model.rc.get(v, None)

        at_bound = "No"
        if val is not None:
            if lb is not None and abs(val - lb) <= tol:
                at_bound = "LB"
            if ub is not None and abs(val - ub) <= tol:
                at_bound = "UB" if at_bound == "No" else "LB/UB"

        lb_str = "-inf" if lb is None else fmt_num(lb)
        ub_str = "+inf" if ub is None else fmt_num(ub)
        rc_str = "N/A" if rc is None else fmt_num(rc)

        print(
            f"{v.name:15s}"
            f"{fmt_num(val):>14s}"
            f"{lb_str:>16s}"
            f"{ub_str:>16s}"
            f"{rc_str:>16s}"
            f"{at_bound:>14s}"
        )


def print_constraint_report(model, tol=1e-8):
    sub_section("Constraint Report")
    header = (
        f"{'Constraint':15s}"
        f"{'Body':>14s}"
        f"{'Lower':>14s}"
        f"{'Upper':>14s}"
        f"{'Slack':>14s}"
        f"{'Binding?':>12s}"
        f"{'Dual':>14s}"
    )
    print(header)
    print(line("-"))

    for c in model.component_data_objects(Constraint, active=True):
        body = safe_value(c.body)
        lb = safe_value(c.lower)
        ub = safe_value(c.upper)

        # For LP reporting, this example focuses on <= constraints,
        # but still handles general bounds.
        slack = None
        binding = "No"

        if ub is not None and body is not None:
            slack = ub - body
            if abs(slack) <= tol:
                binding = "Yes"

        elif lb is not None and body is not None:
            slack = body - lb
            if abs(slack) <= tol:
                binding = "Yes"

        dual = model.dual.get(c, None)

        body_str = fmt_num(body)
        lb_str = "-inf" if lb is None else fmt_num(lb)
        ub_str = "+inf" if ub is None else fmt_num(ub)
        slack_str = "N/A" if slack is None else fmt_num(slack)
        dual_str = "N/A" if dual is None else fmt_num(dual)

        print(
            f"{c.name:15s}"
            f"{body_str:>14s}"
            f"{lb_str:>14s}"
            f"{ub_str:>14s}"
            f"{slack_str:>14s}"
            f"{binding:>12s}"
            f"{dual_str:>14s}"
        )


def print_compact_answer(model):
    sub_section("Optimal Decision")
    print(f"x = {fmt_num(safe_value(model.x))}")
    print(f"y = {fmt_num(safe_value(model.y))}")
    print(f"Objective = {fmt_num(safe_value(model.obj))}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    section("PYOMO LP EXAMPLE WITH GUROBI-LIKE REPORTING")

    # Build model
    model = build_model()

    print_problem_definition()
    print_model_statistics(model)

    # Select solver
    solver_name, solver = choose_solver()
    if solver is None:
        section("ERROR")
        print("No supported solver was found.")
        print("Please install one of the following and ensure it is on PATH:")
        print("  - HiGHS")
        print("  - GLPK")
        print("  - CBC")
        print("  - Gurobi")
        print("  - CPLEX")
        return

    # Solve
    print_solver_header(solver_name)

    # tee=True streams the actual solver log
    results = solver.solve(model, tee=True)

    sub_section("Optimization Complete")
    print("Solver finished execution.")

    print_solver_summary(results, solver_name)

    if not is_optimal(results):
        section("NO OPTIMAL SOLUTION")
        print("The solver did not return an optimal solution.")
        print("Please inspect the solver summary and log above.")
        return

    print_solution_summary(model, results)
    print_variable_report(model)
    print_constraint_report(model)
    print_compact_answer(model)

    section("DONE")
    print("Optimization completed successfully.")
    print("Optimal solution found.")


if __name__ == "__main__":
    main()