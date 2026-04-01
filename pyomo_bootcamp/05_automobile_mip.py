"""Automobile Seat Production Problem

A firm that produces automobile seats manufactures three seat types on two different production
lines. Up to 30 workers can be used at the same time on each line. Each worker is paid:
- $400 per week on production line 1
- $600 per week on production line 2

One week of production setup costs:
- $1,000 on production line 1
- $2,000 on production line 2

The table below provides the seat units that each worker produces in one week on each production
line. The weekly demand is at least:
- 120,000 units of seat 1
- 150,000 units of seat 2
- 200,000 units of seat 3

Production Table ('000 units)
================================
| Production Line | Seat 1 | Seat 2 | Seat 3 |
| --------------- | -----: | -----: | -----: |
| 1               |     20 |     30 |     40 |
| 2               |     50 |     35 |     45 |

Key Cost Data
===============
| Production Line | Worker Cost / Week | Weekly Setup Cost |
| --------------- | ------------------ | ----------------- |
| 1	              |                400 |              1000 |
| 2               |                600 |              2000 |

Formulate an integer linear programming model to minimize the total production cost while meeting
weekly demand.
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
    model.seats = pyo.Set(initialize=["seat1", "seat2", "seat3"])
    model.lines = pyo.Set(initialize=["line1", "line2"])
    seats = model.seats
    lines = model.lines

    # Parameters
    model.setup_cost = pyo.Param(lines, initialize={"line1": 1000, "line2": 2000})
    model.worker_cost = pyo.Param(lines, initialize={"line1": 400, "line2": 600})
    model.production = pyo.Param(lines, seats, initialize={
        ("line1", "seat1"): 20,
        ("line1", "seat2"): 30,
        ("line1", "seat3"): 40,
        ("line2", "seat1"): 50,
        ("line2", "seat2"): 35,
        ("line2", "seat3"): 45,
    })
    model.demand = pyo.Param(seats, initialize={
        "seat1": 120,
        "seat2": 150,
        "seat3": 200,
    })
    model.max_workers = pyo.Param(initialize=30)

    setup_cost = model.setup_cost
    worker_cost = model.worker_cost
    production = model.production
    demand = model.demand
    max_workers = model.max_workers

    # Decision variables
    model.x = pyo.Var(lines, seats, within=pyo.NonNegativeIntegers)
    model.y = pyo.Var(lines, within=pyo.Binary)
    x = model.x
    y = model.y

    # Objective function: Worker Cost + Setup Cost
    def objective_rule(model):
        return sum(x[l, s] * worker_cost[l] for l in lines for s in seats) +\
            sum(y[l] * setup_cost[l] for l in lines)

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    # Seat demand constraint
    def demand_constraint_rule(model, s):
        return sum(x[l, s] * production[l, s] for l in lines) >= demand[s]
    
    model.SeatDemandConstraint = pyo.Constraint(seats, rule=demand_constraint_rule)

    # Max workers constraint
    def max_workers_rule(model, l):
        return sum(x[l, s] for s in seats) <= max_workers * y[l]
    
    model.MaxWorkersConstraint = pyo.Constraint(lines, rule=max_workers_rule)

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
    for l in model.lines:
        for s in model.seats:
            print(f"Workers in {l}, {s}: {model.x[l, s]()}")

    print("=" * 30)

    for l in model.lines:
        total_workers = 0
        for s in model.seats:
            total_workers += model.x[l, s]()
        print(f"Total Workers in {l}: {total_workers}")


if __name__ == "__main__":
    main()
