"""Production Planning NLP Problem

An oil-packing firm contemplates the problem of determining how many units of three product types
it must pack in a given month by considering various constraints. The products in question are
provided in the table below.

Production Planning
======================
| Product          | Packing | Cost of Oil | Production Cost | Market | Sales Price |  Rappel |  Y  |	   LU      |
|                  |         |  (cent/l)   |   (cent/unit)   |   (l)  |  (cent/l)   |    (l)  |     | (Liter/unit) |
| ---------------- | ------- | ----------- | --------------- | ------ | ----------- | ------- | --- | ------------ |
| Virgin Olive Oil | Glass 1 |     345     |       75	     | 175000 |	    445     |  350000 |  8  |	  0.75     |
| Olive oil, 1°    | PVC     |     270     |       15        | 900000 |     303     |  750000 |  4  |	  1.00     |
| Olive oil, 0.4°  | PVC     |	   280     |       15        | 750000 |     303     |  750000 |  4  |     1.00     |

"Production cost" includes packing, labelling, labor, power and others.
"Market" refers to the minimum market share that must be covered.
"Rappel" refers to the linear reduction in the sale price by 1 cent for every X liters produced of
each product.

The cost to store oil is 2 cents per liter per month, which is increased by "Y" cents multiplied
by the percentage of liters of the oil produced. Packaging speed depends on the packaging material:

    - products packed with PVC → 100,000 units/day
    - products packed with glass → 35,000 units/day

The month includes:
20 working days
8-hour shifts
5 employees

Obtain the non-linear programming model of this problem to maximize profit.
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
    model.products = pyo.RangeSet(3)
    products = model.products

    model.pvc_prod = pyo.Set(initialize=[2, 3])
    pvc_prod = model.pvc_prod

    # Parameters
    model.crude_cost_per_l = pyo.Param(products, initialize={
        1: 3.45,
        2: 2.70,
        3: 2.80
    })
    crude_cost_per_l = model.crude_cost_per_l

    model.prod_cost_per_u = pyo.Param(products, initialize={
        1: 0.75,
        2: 0.15,
        3: 0.15
    })
    prod_cost_per_u = model.prod_cost_per_u

    model.store_cost = pyo.Param(initialize=0.02)
    store_cost = model.store_cost

    model.cost_y = pyo.Param(products, initialize={
        1: 0.08,
        2: 0.04,
        3: 0.04,
    })
    cost_y = model.cost_y

    model.demand_l = pyo.Param(products, initialize={
        1: 175000,
        2: 900000,
        3: 750000
    })
    demand_l = model.demand_l

    model.sell_price_per_l = pyo.Param(products, initialize={
        1: 4.45,
        2: 3.03,
        3: 3.03
    })
    sell_price_per_l = model.sell_price_per_l

    model.rappel_l = pyo.Param(products, initialize={
        1: 350000,
        2: 750000,
        3: 750000
    })
    rappel_l = model.rappel_l

    model.l_per_u = pyo.Param(products, initialize={
        1: 0.75,
        2: 1.00,
        3: 1.00
    })
    l_per_u = model.l_per_u

    model.pack_rate_u_per_d = pyo.Param(products, initialize={
        1: 35_000,
        2: 100_000,
        3: 100_000
    })
    pack_rate_u_per_d = model.pack_rate_u_per_d

    model.total_days = pyo.Param(initialize=20)
    total_days = model.total_days

    # Decision variables (units manufactured)
    model.x = pyo.Var(products, within=pyo.NonNegativeIntegers)
    x = model.x

    # Objective function: Worker Cost + Setup Cost
    def objective_rule(model):
        crude_cost = sum(crude_cost_per_l[p] * l_per_u[p] * x[p] for p in products)
        prod_cost = sum(prod_cost_per_u[p] * x[p] for p in products)
        storage_cost = sum((store_cost + cost_y[p] * x[p] * l_per_u[p] / \
                            sum(x[q] * l_per_u[q] for q in products)) * \
                                x[p] * l_per_u[p] for p in products)
        total_cost = crude_cost + prod_cost + storage_cost

        total_revenue = sum((sell_price_per_l[p] - x[p] * l_per_u[p] * 0.01 / rappel_l[p]) * \
                            x[p] * l_per_u[p] for p in products)
        
        profit = total_revenue - total_cost
        return profit

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Constraints
    def market_cover_constraint_rule(model, p):
        return x[p] * l_per_u[p] >= demand_l[p]
    
    model.MarketDemandConstraint = pyo.Constraint(products, rule=market_cover_constraint_rule)

    def max_prod_constraint_rule(model, p):
        return x[p] <= pack_rate_u_per_d[p] * total_days

    model.MaxProdConstraint = pyo.Constraint(products, rule=max_prod_constraint_rule)

    def max_pvc_constraint_rule(model):
        return sum(x[p] for p in pvc_prod) <= 100_000 * total_days
    
    model.MaxPVCConstraint = pyo.Constraint(rule=max_pvc_constraint_rule)
    
    return model


def choose_solver(solver_name):
    """Choose solver. Possible candidates:
        "gurobi", "cplex_direct", "knitroampl", "baron", "highs", "scip", "ipopt",
        "gurobi_direct_minlp"
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
    for p in model.products:
        print(f"Quantity of Product {p}: {model.x[p]()}")

if __name__ == "__main__":
    main()
