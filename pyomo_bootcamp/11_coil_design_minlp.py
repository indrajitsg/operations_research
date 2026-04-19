"""Coil Design MINLP Problem

Consider the problem of designing a coil compression spring which is operating under constant load.
The objective is to design a spring with least material (minimum volume). The spring is to be
helical (spiral) compression spring and the load is considered to be strictly axial. The coil
spring is to be manufactured from music wire spring steel ASTM A228. This means that the wire
diameter can only assume the discrete values shown in a table given in an Excel file. The ends of
the spring are to be squared and ground. The design limitations placed on the design are specified
as:

1. Preload is to be 300 lb †
2. Maximum working load is to be 1000 lb †
3. Maximum allowable deflection under preload is to be 6 in †
4. The deflection from preload position to maximum load position is to be 1.25 in †
5. The maximum free length of the coil spring is to be 14 in †
6. The maximum outside diameter of the spring is to be 3 in †
7. The minimum wire diameter of the spring is to be 0.20 in †
8. The end coefficient is to be 1.0 †

Additional data:
==================
Max Sheer Stress (S) = 234.44e3
Sheer Modulus ASTM a228 (G) = 11.6e6
d → Wire diameter
D → Mean coil diameter
N → Number of active coils
Pmax → Max working load

Volume (Objective) = π * D * d^2 * (N + 2)/4

C = D / d
K = (4*C - 1)/(4*C - 4) + 0.615/C
Stress (S) = (8 * K * Pmax * D) / (π * d^3)

Deflection (def) = (8 * D^3 * N) / (G * d^4)

Free Length (L) = Pmax * def + 1.05 * end_coeff * (N + 2) * d
"""
import os
import sys
import math
import logging
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

os.environ['NEOS_EMAIL'] = 'indrajitsg@gmail.com'

# Set the pyomo.neos logger to INFO level
logging.getLogger('pyomo.neos').setLevel(logging.INFO)
# Standard output handler to ensure it prints to your terminal
logging.basicConfig(level=logging.INFO)


def print_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_coil_data():
    df = pd.read_csv("S7P2_Data.csv")
    wire_types_lst = df["wire_type"].to_list()
    s7p2_dict = dict()
    for idx, row in df.iterrows():
        s7p2_dict.update({int(row["wire_type"]): float(row["diameter"])})
    return s7p2_dict, wire_types_lst


def build_model():
    # Defining model
    model = pyo.ConcreteModel()
    wire_diameters, wire_types_lst = load_coil_data()

    # ==============================
    # Set (Wire type)
    # ==============================
    model.wire_types = pyo.Set(initialize=wire_types_lst)
    wire_types = model.wire_types

    # ==============================
    # Parameters
    # ==============================
    model.dw = pyo.Param(wire_types, initialize=wire_diameters)
    dw = model.dw

    model.max_stress = pyo.Param(initialize=234.44e3)
    max_stress = model.max_stress

    model.G = pyo.Param(initialize=11.6e6)
    G = model.G

    model.Pmax = pyo.Param(initialize=1000)
    Pmax = model.Pmax

    model.Pload = pyo.Param(initialize=300)
    Pload = model.Pload

    model.max_free_len = pyo.Param(initialize=14)
    max_free_len = model.max_free_len

    model.preload_def_len = pyo.Param(initialize=6)
    preload_def_len = model.preload_def_len

    model.min_wire_diameter = pyo.Param(initialize=0.2)
    min_wire_diameter = model.min_wire_diameter

    model.preload_to_max_def_len = pyo.Param(initialize=1.25)
    preload_to_max_def_len = model.preload_to_max_def_len

    model.max_outside_diameter = pyo.Param(initialize=3)
    max_outside_diameter = model.max_outside_diameter

    model.spring_idx_lb = pyo.Param(initialize=4)
    spring_idx_lb = model.spring_idx_lb

    model.end_coeff = pyo.Param(initialize=1.0)
    end_coeff = model.end_coeff

    # ==============================
    # Decision variables
    # ==============================
    model.x = pyo.Var(wire_types, within=pyo.Binary)
    x = model.x

    model.d = pyo.Var(within=pyo.NonNegativeReals)
    d = model.d

    model.D = pyo.Var(within=pyo.NonNegativeReals)
    D = model.D

    model.N = pyo.Var(within=pyo.NonNegativeIntegers)
    N = model.N

    # ==============================
    # Objective function: 
    # Vol = π * D * d^2 * (N + 2)/4
    # ==============================
    def objective_rule(model):
        volume = math.pi * D * d**2 * (N + 2)/4
        return volume

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # ==============================
    # Constraints
    # ==============================
    def wire_diameter_link_rule(model):
        return d == sum(dw[i] * x[i] for i in wire_types)
    
    model.WireDiameterLinkConstraint = pyo.Constraint(rule=wire_diameter_link_rule)

    def max_allowed_stress_rule(model):
        C = D / d
        K = (4*C - 1)/(4*C - 4) + 0.615/C
        stress = (8 * K * Pmax * D) / (math.pi * d**3)
        return stress <= max_stress
    
    model.MaxAllowedStressConstraint = pyo.Constraint(rule=max_allowed_stress_rule)

    def single_wire_type_rule(model):
        return sum(x[i] for i in wire_types) == 1

    model.SingleWireConstraint = pyo.Constraint(rule=single_wire_type_rule)
    
    def max_load_deflection_rule(model):
        compliance = (8 * D**3 * N) / (G * d**4)
        free_length = Pmax * compliance + 1.05 * end_coeff * (N + 2) * d
        return free_length <= max_free_len
    
    model.MaxFreeLengthConstraint = pyo.Constraint(rule=max_load_deflection_rule)

    def preload_deflection_rule(model):
        compliance = (8 * D**3 * N) / (G * d**4)
        deflection = Pload * compliance
        return deflection <= preload_def_len
    
    model.MaxPreloadDeflectionConstraint = pyo.Constraint(rule=preload_deflection_rule)

    def min_wire_diameter_rule(model):
        return d >= min_wire_diameter
    
    model.MinWireDiameterConstraint = pyo.Constraint(rule=min_wire_diameter_rule)

    def preload_to_max_def_rule(model):
        compliance = (8 * D**3 * N) / (G * d**4)
        deflection_preload = Pload * compliance
        deflection_max = Pmax * compliance
        return deflection_max - deflection_preload == preload_to_max_def_len
    
    model.PreLoadToMaxDefConstraint = pyo.Constraint(rule=preload_to_max_def_rule)

    model.PositiveNumCoilsConstraint = pyo.Constraint(expr = N >= 1)

    def max_outside_coil_diameter_rule(model):
        return D + d <= max_outside_diameter

    model.MaxOutsideCoilDiameterConstraint = pyo.Constraint(rule=max_outside_coil_diameter_rule)

    def spring_idx_lb_rule(model):
        return D >= spring_idx_lb * d
    
    model.SpringIdxLowerBound = pyo.Constraint(rule=spring_idx_lb_rule)

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


def create_neos_manager():
    """Create NEOS Solver Manager"""
    solver_manager = pyo.SolverManagerFactory("neos")
    return solver_manager


def print_post_solution_audit(model):
    # Selected wire diameter
    selected_wire = None
    selected_wire_type = None
    for w in model.wire_types:
        x_val = pyo.value(model.x[w])
        if x_val is not None and x_val > 0.5:
            selected_wire = pyo.value(model.dw[w])
            selected_wire_type = w
            break

    if selected_wire is None:
        print("\nPOST-SOLUTION AUDIT")
        print("=" * 70)
        print("No wire type appears to be selected.")
        return

    d = pyo.value(model.d)
    D = pyo.value(model.D)
    N = pyo.value(model.N)

    Pmax = pyo.value(model.Pmax)
    Pload = pyo.value(model.Pload)
    G = pyo.value(model.G)
    max_stress = pyo.value(model.max_stress)
    max_free_len = pyo.value(model.max_free_len)
    preload_def_len = pyo.value(model.preload_def_len)
    preload_to_max_def_len = pyo.value(model.preload_to_max_def_len)
    max_outside_diameter = pyo.value(model.max_outside_diameter)
    spring_idx_lb = pyo.value(model.spring_idx_lb)

    # Derived quantities
    C = D / d
    K = (4 * C - 1) / (4 * C - 4) + 0.615 / C
    compliance = (8 * D**3 * N) / (G * d**4)

    deflection_preload = Pload * compliance
    deflection_max = Pmax * compliance
    added_deflection = deflection_max - deflection_preload
    free_length = deflection_max + 1.05 * (N + 2) * d
    outside_diameter = D + d
    stress = (8 * K * Pmax * D) / (math.pi * d**3)
    volume = math.pi * D * d**2 * (N + 2) / 4

    print("\n" + "=" * 70)
    print("POST-SOLUTION AUDIT")
    print("=" * 70)

    print(f"Selected wire type: {selected_wire_type}")
    print(f"Wire diameter d: {d:.6f}")
    print(f"Mean coil diameter D: {D:.6f}")
    print(f"Active coils N: {N:.6f}")
    print(f"Objective volume: {volume:.6f}")

    print("\nDerived values")
    print(f"Spring index C = D/d: {C:.6f}")
    print(f"Wahl factor K: {K:.6f}")
    print(f"Compliance: {compliance:.10f}")

    print("\nConstraint checks")
    print(
        f"Stress at max load: {stress:.6f} "
        f"<= {max_stress:.6f}   "
        f"[{'OK' if stress <= max_stress + 1e-6 else 'VIOLATION'}]"
    )
    print(
        f"Preload deflection: {deflection_preload:.6f} "
        f"<= {preload_def_len:.6f}   "
        f"[{'OK' if deflection_preload <= preload_def_len + 1e-6 else 'VIOLATION'}]"
    )
    print(
        f"Added deflection (Pload to Pmax): {added_deflection:.6f} "
        f"== {preload_to_max_def_len:.6f}   "
        f"[{'OK' if abs(added_deflection - preload_to_max_def_len) <= 1e-6 else 'VIOLATION'}]"
    )
    print(
        f"Free length: {free_length:.6f} "
        f"<= {max_free_len:.6f}   "
        f"[{'OK' if free_length <= max_free_len + 1e-6 else 'VIOLATION'}]"
    )
    print(
        f"Outside diameter D + d: {outside_diameter:.6f} "
        f"<= {max_outside_diameter:.6f}   "
        f"[{'OK' if outside_diameter <= max_outside_diameter + 1e-6 else 'VIOLATION'}]"
    )
    print(
        f"Minimum wire diameter: {d:.6f} "
        f">= {pyo.value(model.min_wire_diameter):.6f}   "
        f"[{'OK' if d >= pyo.value(model.min_wire_diameter) - 1e-6 else 'VIOLATION'}]"
    )
    print(
        f"Spring index lower bound: {C:.6f} "
        f">= {spring_idx_lb:.6f}   "
        f"[{'OK' if C >= spring_idx_lb - 1e-6 else 'VIOLATION'}]"
    )


def main():
    # Build model
    model = build_model()

    # Select solver
    neos_flag = False

    if len(sys.argv) > 1:
        user_solver_name = sys.argv[1]
        print(f"Solver provided: {user_solver_name}")
    else:
        print("No solver was specified, defaulting to Highs")
        user_solver_name = "highs"
    
    if user_solver_name != "neos":
        solver_name, solver = choose_solver(user_solver_name)
    else:
        # Acceptable solvers: ['baron', 'bonmin', 'cbc', 'conopt', 'couenne', 'cplex', 'filmint',
        # 'filter', 'ipopt', 'knitro', 'l-bfgs-b', 'lancelot', 'lgo', 'loqo', 'minlp', 'minos',
        # 'minto', 'mosek', 'ooqp', 'path', 'raposa', 'snopt']
        neos_flag = True
        solver_name = sys.argv[2]
        # print(f"Using NEOS Solver {solver_name}")
        solver_manager = create_neos_manager()

    # Solve
    print_banner("STARTING SOLVE")
    print(f"Attempting to solve with solver: {solver_name}")
    if not neos_flag:
        results = solver.solve(model, tee=True)
    else:
        action_handle = solver_manager.queue(model, opt=solver_name)

        # Wait for job to finish
        results = solver_manager.wait_for(action_handle)

    # Display solution
    print_banner("RESULTS")
    print(results)
    print(f"Objective: {model.Obj()}")
    print(f"Wire diameter: {model.d()}")
    print(f"Coil Diameter: {model.D()}")
    print(f"Number of coils: {model.N()}")
    
    # Perform post solution audit
    print_post_solution_audit(model)


if __name__ == "__main__":
    main()
