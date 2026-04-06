"""Traveling Salesman Problem (TSP)

Joe State lives in Gary, Indiana. He owns insurance agencies in Gary, Fort Wayne, Evansville,
Terre Haute, and South Bend. Each December, he visits each of his insurance agencies. The distance
between each agency (in miles) is given in an Excel-style table (shown below). What order of
visiting his agencies will minimize the total distance traveled?

Suppose the TSP consists of cities: 1, 2, 3, …, 𝑁

Distance Matrix (miles)
==========================
| From / To | City1 | City2 | City3 | City4 | City5 |
| --------- | ----: | ----: | ----: | ----: | ----: |
|   City1   |     0 |   132 |   217 |   164 |    58 |
|   City2   |   132 |     0 |   290 |   201 |    79 |
|   City3   |   217 |   290 |     0 |   113 |   303 |
|   City4   |   164 |   201 |   113 |     0 |   196 |
|   City5   |    58 |    79 |   303 |   196 |     0 |
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
    model.cities = pyo.RangeSet(5)
    model.non_depot = pyo.RangeSet(2, 5)
    cities = model.cities
    non_depot = model.non_depot

    # Parameters
    model.distance = pyo.Param(cities, cities, initialize={
        (1, 1): 0, (1, 2): 132, (1, 3): 217, (1, 4): 164, (1, 5): 58,
        (2, 1): 132, (2, 2): 0, (2, 3): 290, (2, 4): 201, (2, 5): 79,
        (3, 1): 217, (3, 2): 290, (3, 3): 0, (3, 4): 113, (3, 5): 303,
        (4, 1): 164, (4, 2): 201, (4, 3): 113, (4, 4): 0, (4, 5): 196,
        (5, 1): 58, (5, 2): 79, (5, 3): 303, (5, 4): 196, (5, 5): 0
    })
    distance = model.distance

    model.N = pyo.Param(initialize=5)
    N = model.N

    # Decision variables
    model.x = pyo.Var(cities, cities, within=pyo.Binary)
    model.u = pyo.Var(cities, within=pyo.NonNegativeIntegers, bounds=(1, len(cities)))
    x = model.x
    u = model.u  # For assigning a rank - which prevents multiple subtours

    # Objective function: Worker Cost + Setup Cost
    def objective_rule(model):
        return sum(x[i, j] * distance[i, j] for i in cities for j in cities)

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    # Only one route to enter a city
    def entry_constraint_rule(model, j):
        return sum(x[i, j] for i in cities) == 1
    
    model.CityEntryConstraint = pyo.Constraint(cities, rule=entry_constraint_rule)

    # Only one route to exit a city
    def exit_constraint_rule(model, i):
        return sum(x[i, j] for j in cities) == 1
    
    model.CityExitConstraint = pyo.Constraint(cities, rule=exit_constraint_rule)

    # No self loop
    def loop_constraint_rule(model, i):
        return x[i, i] == 0

    model.NoLoopConstraint = pyo.Constraint(cities, rule=loop_constraint_rule)

    # First city rank constraint
    def first_city_constraint_rule(model):
        return u[1] == 1
    
    model.FirstCityConstraint = pyo.Constraint(rule=first_city_constraint_rule)

    # Non-depot rank constraint
    def non_depot_constraint_rule(model, i):
        return u[i] >= 2
    
    model.NonDepotConstraint = pyo.Constraint(non_depot, rule=non_depot_constraint_rule)

    # No sub-tours
    valid_pairs = [(i, j) for i in non_depot for j in non_depot if i != j]
    def subtour_constraint_rule(model, i, j):
        return u[i] - u[j] + N * x[i, j] <= N - 1
    
    model.NoSubTourConstraint = pyo.Constraint(valid_pairs, rule=subtour_constraint_rule)
    
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
    for i in model.cities:
        for j in model.cities:
            if model.x[i, j]() > 0:
                print(f"City {i} -> City {j}")

    for j in model.cities:
        print(f"u[{j}]: {model.u[j]()}")

if __name__ == "__main__":
    main()
