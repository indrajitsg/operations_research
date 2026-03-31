"""Dakota Furniture LP Problem

The Dakota Furniture Company manufactures desks, tables, and chairs. The manufacture of each type
of furniture requires lumber and two types of skilled labor: finishing and carpentry. The amount of
each resource needed to make each type of furniture is given in the below Table. Currently, 48
board feet of lumber, 20 finishing hours, and 8 carpentry hours are available. A desk sells for
$60, a table for $30, and a chair for $20. Dakota believes that demand for desks and chairs is
unlimited, but at most five tables can be sold. Because the available resources have already been
purchased, Dakota wants to maximize total revenue.

| Resource  | Desk | Table | Chair | Available |
| --------- | ---- | ----- | ----- | --------- |
| Lumber    | 8    | 6     | 1     | 48        |
| Finishing | 4    | 2     | 1.5   | 20        |
| Carpentry | 2    | 1.5   | 0.5   | 8         |
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
    model.products = pyo.Set(initialize=["Desk", "Table", "Chair"])
    products = model.products

    # Parameters
    model.lumber = pyo.Param(products, initialize= {"Desk": 8, "Table": 6, "Chair": 1})
    model.finishing = pyo.Param(products, initialize= {"Desk": 4, "Table": 2, "Chair": 1.5})
    model.carpentry = pyo.Param(products, initialize= {"Desk": 2, "Table": 1.5, "Chair": 0.5})
    model.price = pyo.Param(products, initialize= {"Desk": 60, "Table": 30, "Chair": 20})
    lumber = model.lumber
    finishing = model.finishing
    carpentry = model.carpentry
    price = model.price

    # Decision variables
    model.x = pyo.Var(products, within=pyo.NonNegativeIntegers)
    x = model.x

    # Objective function
    def objective_rule(model):
        return sum(price[p] * x[p] for p in model.products)

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    # Lumber availability: 48 board feet available
    def lumber_constraint_rule(model):
        return sum(lumber[p] * x[p] for p in model.products) <= 48

    model.LumberConstraint = pyo.Constraint(rule=lumber_constraint_rule)

    # Finishing availability 20 hours
    def finishing_constraint_rule(model):
        return sum(finishing[p] * x[p] for p in model.products) <= 20
    
    model.FinishingConstraint = pyo.Constraint(rule=finishing_constraint_rule)

    # Carpentry availability 8 hours
    def carpentry_constraint_rule(model):
        return sum(carpentry[p] * x[p] for p in model.products) <= 8
    
    model.CarpentryConstraint = pyo.Constraint(rule=carpentry_constraint_rule)

    # Demand Constraint for "Table"
    model.TableDemandConstraint = pyo.Constraint(
        expr=x["Table"] <= 5
    )

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
    results = solver.solve(model)

    # Display solution
    print_banner("RESULTS")
    print(results)
    print(f"Objective: {model.Obj()}")
    for p in model.products:
        print(f"{p} = {model.x[p]()}")


if __name__ == "__main__":
    main()

