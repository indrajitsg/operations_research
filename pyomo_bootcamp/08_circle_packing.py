"""Circular Object Packing Problem

A firm must design a box of minimum dimensions to pack three circular objects with the following
radii:

R1 = 6 cm
R2 = 12 cm
R3 = 16 cm

By bearing in mind that the circles placed inside the box cannot overlap, consider a non-linear
programming model that minimizes the perimeter of the box.
"""
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def draw_rectangle(ax, width, height):
    """
    Draw the bounding rectangle (box).
    Bottom-left corner is assumed at (0,0).
    """
    rect = Rectangle((0, 0), width, height, fill=False, linewidth=2)
    ax.add_patch(rect)


def draw_circle(ax, x, y, radius, label=None):
    """
    Draw a single circle inside the rectangle.
    """
    circle = Circle((x, y), radius, fill=False, linewidth=2)
    ax.add_patch(circle)

    if label is not None:
        ax.text(x, y, str(label), ha="center", va="center")


def draw_packing_solution(box_width, box_height, circles):
    """
    Draw the full packing solution.

    Parameters
    ----------
    box_width : float
    box_height : float
    circles : list of tuples
        Each tuple = (x_center, y_center, radius, label)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw rectangle
    draw_rectangle(ax, box_width, box_height)

    # Draw all circles
    for x, y, r, label in circles:
        draw_circle(ax, x, y, r, label)

    # Formatting
    ax.set_xlim(0, box_width + 5)
    ax.set_ylim(0, box_height + 5)
    ax.set_aspect("equal")
    ax.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Circle Packing Solution")
    plt.show()



def print_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_model():
    # Defining model
    model = pyo.ConcreteModel()

    # Sets
    model.circles = pyo.RangeSet(3)
    circles = model.circles

    # Parameters
    model.radii = pyo.Param(circles, initialize={
        1: 6, 2: 12, 3: 16
    })
    radii = model.radii

    # Decision variables
    model.x = pyo.Var(circles, within=pyo.NonNegativeReals, bounds=(0, 100))
    model.y = pyo.Var(circles, within=pyo.NonNegativeReals, bounds=(0, 100))
    model.a = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 100))
    model.b = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 100))

    x = model.x
    y = model.y
    a = model.a
    b = model.b

    # Objective function: Minimize the perimeter of the rectangle
    def objective_rule(model):
        return 2 * (a + b)

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints: every center is inside the box
    def x_axis_pos_constraint_rule(model, c):
        return x[c] + radii[c] <= a
    
    model.XAxisPosConstraint = pyo.Constraint(circles, rule=x_axis_pos_constraint_rule)
    
    def x_axis_neg_constraint_rule(model, c):
        return x[c] - radii[c] >= 0
    
    model.XAxisNegConstraint = pyo.Constraint(circles, rule=x_axis_neg_constraint_rule)

    def y_axis_pos_constraint_rule(model, c):
        return y[c] + radii[c] <= b

    model.YAxisPosConstraint = pyo.Constraint(circles, rule=y_axis_pos_constraint_rule)

    def y_axis_neg_constraint_rule(model, c):
        return y[c] - radii[c] >= 0
    
    model.YAxisNegConstraint = pyo.Constraint(circles, rule=y_axis_neg_constraint_rule)

    # Constraint: No Overlap
    subset_circles = [(c1, c2) for c1 in circles for c2 in circles if c1 < c2]

    def no_overlap_constraint_rule(model, c1, c2):
        dist_sq = (y[c1] - y[c2])**2 + (x[c1] - x[c2])**2
        return dist_sq >= (radii[c1] + radii[c2])**2
            
    model.NoOverlapConstraint = pyo.Constraint(subset_circles, rule=no_overlap_constraint_rule)

    return model


def choose_solver(solver_name):
    """Choose solver. Possible candidates:
        "gurobi", "cplex_direct", "knitroampl", "baron", "highs", "scip", "ipopt"
    """
    try:
        solver = SolverFactory(solver_name)
        if solver_name == "knitroampl":
            solver.options["par_numthreads"] = 2
            solver.options["ms_enable"] = 1
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
    print(f"Width of Rectangle: {model.a()}")
    print(f"Height of Rectangle: {model.b()}")

    circle_tup = []
    for c in model.circles:
        print(f"Location of Circle {c}: {(model.x[c](), model.y[c]())}")
        circle_tup.append((model.x[c](), model.y[c](), model.radii[c], "C"+str(c)))

    # print(circle_tup)    
    draw_packing_solution(
        box_width=model.a(),
        box_height=model.b(),
        circles=circle_tup
    )


if __name__ == "__main__":
    main()

