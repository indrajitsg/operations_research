"""Gandhi Cloth Company MIP Problem

Gandhi Cloth Company is capable of manufacturing three types of clothing:

1. shirts
2. shorts
3. pants

The manufacture of each type of clothing requires that Gandhi have the appropriate type of
machinery available. The machinery needed to manufacture each type of clothing must be rented at
the following weekly rates:

1. shirt machinery: $200 per week
2. shorts machinery: $150 per week
3. pants machinery: $100 per week

The manufacture of each type of clothing also requires cloth and labor as given in the table below.
Each week, the following resources are available:

1. 150 hours of labor
2. 160 sq yd of cloth

The variable unit cost and selling price for each type of clothing are shown below. The goal is to:
Formulate an Integer Program (IP) whose solution maximizes Gandhi’s weekly profits.

Product Data
==============
| Type   | Sales Price ($) | Cost ($) | Labor (hours) | Cloth (sq yd) |
| ------ | --------------- | -------- | ------------- | ------------- |
| Shirt  | 12              | 6        | 3             | 4             |
| Shorts | 8               | 4        | 2             | 3             |
| Pants  | 15              | 8        | 6             | 4             |

Fixed Machinery Rental Costs
=============================
| Product | Weekly Machine Rental Cost |
| ------- | -------------------------- |
| Shirt   | 200                        |
| Shorts  | 150                        |
| Pants   | 100                        |
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
    model.products = pyo.Set(initialize=["shirts", "shorts", "pants"])
    products = model.products

    # Parameters
    model.fixed_cost = pyo.Param(products, initialize={"shirts": 200, "shorts": 150, "pants": 100})
    model.cloth = pyo.Param(products, initialize={"shirts": 4, "shorts": 3, "pants": 4})
    model.labor = pyo.Param(products, initialize={"shirts": 3, "shorts": 2, "pants": 6})
    model.cost = pyo.Param(products, initialize={"shirts": 6, "shorts": 4, "pants": 8})
    model.price = pyo.Param(products, initialize={"shirts": 12, "shorts": 8, "pants": 15})
    model.total_labor = pyo.Param(initialize=180) # 150
    model.total_cloth = pyo.Param(initialize=190) # 160
    model.M = pyo.Param(initialize=1000)

    fixed_cost = model.fixed_cost
    cloth = model.cloth
    labor = model.labor
    cost = model.cost
    price = model.price
    total_labor = model.total_labor
    total_cloth = model.total_cloth
    M = model.M

    # Decision variables
    model.x = pyo.Var(products, within=pyo.NonNegativeIntegers)
    model.y = pyo.Var(products, within=pyo.Binary)
    x = model.x
    y = model.y

    # Objective function
    def objective_rule(model):
        return sum(x[p] * price[p] - x[p] * cost[p] - y[p] * fixed_cost[p] for p in products)

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    # Labor hours constraint
    def labor_constraint_rule(model):
        return sum(x[p] * labor[p] for p in products) <= total_labor
    
    model.LaborHoursConstraint = pyo.Constraint(rule=labor_constraint_rule)

    # Cloth supply constraint
    def cloth_constraint_rule(model):
        return sum(x[p] * cloth[p] for p in products) <= total_cloth
    
    model.ClothSupplyConstraint = pyo.Constraint(rule=cloth_constraint_rule)

    # Big M constraint
    def bigm_constraint_rule(model, p):
        return x[p] <= M * y[p]
    
    model.BigMConstraint = pyo.Constraint(products, rule=bigm_constraint_rule)

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
    for p in model.products:
        print(f"No. of {p}: {model.x[p]()}")

if __name__ == "__main__":
    main()
