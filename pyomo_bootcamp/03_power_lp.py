"""Powerco Electric LP Problem

Powerco has three electric power plants that supply the needs of four cities. Each power plant can
supply the following numbers of kilowatt-hours (kwh) of electricity: plant 1— 35 million; plant
2—50 million; plant 3— 40 million (See below Table). The peak power demands in these cities, which
occur at the same time (2 P.M.), are as follows (in kwh): city 1—45 million; city 2—20 million;
city 3—30 million; city 4—30 million. The costs of sending 1 million kwh of electricity from plant
to city depend on the distance the electricity must travel. Formulate an LP to minimize the cost of
meeting each city’s peak power demand.

Costs (per 1 million kWh)
==========================
| From \ To | City1 | City2 | City3 | City4 | Supply (Million kWh) |
| --------- | ----- | ----- | ----- | ----- | -------------------- |
| Plant1    | 8     | 6     | 10    | 9     | 35                   |
| Plant2    | 9     | 12    | 13    | 7     | 50                   |
| Plant3    | 14    | 9     | 16    | 5     | 40                   |

Demand (Million kWh)
| City  | Demand |
| ----- | ------ |
| City1 | 45     |
| City2 | 20     |
| City3 | 30     |
| City4 | 30     |
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
    model.cities = pyo.Set(initialize=["City1", "City2", "City3", "City4"])
    model.plants = pyo.Set(initialize=["Plant1", "Plant2", "Plant3"])
    cities = model.cities
    plants = model.plants

    # Parameters
    model.demand = pyo.Param(cities, initialize={"City1": 45, "City2": 20, "City3": 30, "City4": 30})
    model.supply = pyo.Param(plants, initialize={"Plant1": 35, "Plant2": 50, "Plant3": 40})
    model.costs = pyo.Param(plants, cities,
                            initialize={
                                ("Plant1", "City1"): 8,
                                ("Plant1", "City2"): 6,
                                ("Plant1", "City3"): 10,
                                ("Plant1", "City4"): 9,
                                ("Plant2", "City1"): 9,
                                ("Plant2", "City2"): 12,
                                ("Plant2", "City3"): 13,
                                ("Plant2", "City4"): 7,
                                ("Plant3", "City1"): 14,
                                ("Plant3", "City2"): 9,
                                ("Plant3", "City3"): 16,
                                ("Plant3", "City4"): 5,
                            })
    demand = model.demand
    supply = model.supply
    costs = model.costs

    # Decision variables
    model.x = pyo.Var(plants, cities, within=pyo.NonNegativeReals)
    x = model.x

    # Objective function
    def objective_rule(model):
        return sum(costs[p, c] * x[p, c] for p in plants for c in cities)

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    # Demand Constraint for each city
    def demand_constraint_rule(model, c):
        return sum(x[p, c] for p in plants) == demand[c]
    
    model.PowerDemandConstraint = pyo.Constraint(cities, rule=demand_constraint_rule)

    # Supply Constraint for each plant
    def supply_constraint_rule(model, p):
        return sum(x[p, c] for c in cities) <= supply[p]
    
    model.PowerSupplyConstraint = pyo.Constraint(plants, rule=supply_constraint_rule)
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
    for p in model.plants:
        for c in model.cities:
            output = model.x[p, c]()
            if output > 0:
                print(f"From {p} -> {c} = {output}")


if __name__ == "__main__":
    main()

