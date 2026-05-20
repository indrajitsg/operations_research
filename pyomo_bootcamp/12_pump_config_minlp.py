"""Challenge Problem! Pump Configuration

The aim is to identify the least costly configuration of centrifugal pumps that achieves a 
pre-specified pressure rise based on a given total flowrate. The structural decisions for an L
level superstructure are represented by a number of discrete variables. The binary variable
z_i (i=1, ..., L) denotes the existence of level i. The integer variables N_p_i denotes the number
of parallel lines at level i, while the N_s_i denote the number of pumps in series at level i
(number of pumps in each line). The relevant continuous variables are the fraction of total flow
going to level i (x_i), the power requirement for a pump at level i (P_i), and the pressure rise at
level i (Δp_i). For a 3 level structure, formulate a MINLP problem to minimize the annualized cost.

Level: Set (i = 1, 2, 3)

Parameters:
------------------------------------------------------------------------------------
           C        C'    alpha    beta     gammma     a      b         c     Pmax
------------------------------------------------------------------------------------
Level1   6329.03   1800   19.90   0.1610   -0.00056   629   0.696   -0.01160   80
Level2   2489.31   1800    1.21   0.0644   -0.00056   215   2.950   -0.11500   25
Level3   3270.27   1800    6.52   0.1020   -0.00023   361   0.530   -0.00946   35
------------------------------------------------------------------------------------

C : Yearly installment of the fixed cost
C': Power Cost
alpha - c: Coefficients
Pmax: Maximum power of each pump

Binary Variables:
z_i: denotes the existence of level i

Positive Integer Variables:
N_p_i: number of parallel lines at level i
N_s_i: number of pumps in series at level i

Positive Real Variables:
x_i: fraction of total flow going to level i
w_i: rotation speeed of all pumps at level i
v_i: flowrate on each line at level i
P_i: power requirements at level i
Δp_i: pressure rise at level i

Obj: Min Cost = (Pump Fixed Cost + Elec Cost x Power Cons) x # of Lines x # of Pumps per Line
=> Min Cost = ∑_i (C_i + C'_i x P_i) x N_p_i x N_s_i x z_i

Constraints:
1. Sum x_i (for all i) == 1 †
2. P_i - alpha_i * (w_i/w_max)^3 - beta_i * (w_i / w_max)^2 * v_i - gamma_i * (w_i / w_max) * v_i^2 = 0 †
3. Δp_i - a_i * (w_i / w_max)^2 - b_i * (w_i / w_max) * v_i - c_i * v_i^2 = 0 †
4. v_i * N_p_i - x_i * V_tot = 0 †
5. ΔP_tot * z_i - Δp_i * N_s_i = 0 †
6. 0 <= x_i <= 1 †
7. 0 <= v_i <= V_tot †
8. 0 <= w_i <= w_max †
9. N_p_i ∈ {0, 1, 2, 3} †
10. N_s_i ∈ {0, 1, 2, 3} †
11. 0 <= Δp_i <= ΔP_tot †
12. 0 <= P_i <= P_max_i †
13. z_i ∈ {0, 1} †

Additional constraints to reduce the problem:
14. P_i <= z_i * P_max_i †
15. Δp_i <= z_i * ΔP_tot †
16. v_i <= z_i * V_tot †
17. w_i <= z_i * w_max †
18. N_p_i <= z_i * NP_max †
19. N_s_i <= z_i * NS_max †
20. x_i <= z_i †
21. N_p_i >= z_i †
22. N_s_i >= z_i †
"""
import os
import sys
import logging
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

os.environ['NEOS_EMAIL'] = 'indrajitsg@gmail.com'

# Set the pyomo.neos logger to INFO level
logging.getLogger('pyomo.neos').setLevel(logging.INFO)


def print_banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def create_param_dict(param_lst):
    param_dict = dict()
    level = 1
    for param in param_lst:
        param_dict.update({level: param})
        level += 1
    return param_dict


def build_model():
    # Defining model
    model = pyo.ConcreteModel()

    # ==============================
    # Sets
    # ==============================
    model.levels = pyo.Set(initialize=[1, 2, 3])
    levels = model.levels

    # ==============================
    # Parameters
    # ==============================
    pump_df = pd.read_csv("pump_data.csv")

    model.fixed_cost = pyo.Param(levels, initialize=create_param_dict(list(pump_df["cost"])))
    fixed_cost = model.fixed_cost

    model.power_cost = pyo.Param(levels, initialize=create_param_dict(list(pump_df["c_prime"])))
    power_cost = model.power_cost

    model.alpha = pyo.Param(levels, initialize=create_param_dict(list(pump_df["alpha"])))
    alpha = model.alpha

    model.beta = pyo.Param(levels, initialize=create_param_dict(list(pump_df["beta"])))
    beta = model.beta

    model.gamma = pyo.Param(levels, initialize=create_param_dict(list(pump_df["gamma"])))
    gamma = model.gamma

    model.a = pyo.Param(levels, initialize=create_param_dict(list(pump_df["a"])))
    a = model.a

    model.b = pyo.Param(levels, initialize=create_param_dict(list(pump_df["b"])))
    b = model.b

    model.c = pyo.Param(levels, initialize=create_param_dict(list(pump_df["c"])))
    c = model.c

    model.pmax = pyo.Param(levels, initialize=create_param_dict(list(pump_df["pmax"])))
    pmax = model.pmax

    model.w_max = pyo.Param(initialize=2950)
    w_max = model.w_max

    model.v_tot = pyo.Param(initialize=350)
    v_tot = model.v_tot

    model.delta_p_tot = pyo.Param(initialize=400)
    delta_p_tot = model.delta_p_tot

    model.Np_max = pyo.Param(initialize=3)
    Np_max = model.Np_max

    model.Ns_max = pyo.Param(initialize=3)
    Ns_max = model.Ns_max

    # ==============================
    # Decision variables
    # ==============================
    model.z = pyo.Var(levels, within=pyo.Binary)
    z = model.z

    model.Np = pyo.Var(levels, within=pyo.NonNegativeIntegers, bounds=[0, 3])
    Np = model.Np

    model.Ns = pyo.Var(levels, within=pyo.NonNegativeIntegers, bounds=[0, 3])
    Ns = model.Ns

    model.x = pyo.Var(levels, within=pyo.NonNegativeReals, bounds=[0, 1])
    x = model.x

    model.w = pyo.Var(levels, within=pyo.NonNegativeReals, bounds=[0, w_max])
    w = model.w

    model.v = pyo.Var(levels, within=pyo.NonNegativeReals, bounds=[0, v_tot])
    v = model.v

    model.P = pyo.Var(levels, within=pyo.NonNegativeReals)
    P = model.P

    model.delta_p = pyo.Var(levels, within=pyo.NonNegativeReals, bounds=[0, delta_p_tot])
    delta_p = model.delta_p

    # ==============================
    # Objective function: 
    # Min Cost = ∑_i (C_i + C'_i x P_i) x N_p_i x N_s_i x z_i
    # ==============================
    def objective_rule(model):
        total_cost = sum((fixed_cost[l] + power_cost[l] * P[l]) * Np[l] * Ns[l] for l in levels) # Removed  * z[l]
        return total_cost

    model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # ==============================
    # Constraints
    # ==============================

    # 1. Sum x_i (for all i) == 1
    def cons1_rule(model):
        return sum(x[l] for l in levels) == 1
    
    model.Constraint1 = pyo.Constraint(rule=cons1_rule)

    # 2. P_i - alpha_i * (w_i/w_max)^3 - beta_i * (w_i / w_max)^2 * v_i - gamma_i * (w_i / w_max) * v_i^2 = 0
    def cons2_rule(model, l):
        term1 = P[l]
        term2 = alpha[l] * (w[l] / w_max)**3
        term3 = beta[l] * (w[l] / w_max)**2 * v[l]
        term4 = gamma[l] * (w[l] / w_max) * v[l]**2
        return term1 - term2 - term3 - term4 == 0
    
    model.Constraint2 = pyo.Constraint(levels, rule=cons2_rule)

    # 3. Δp_i - a_i * (w_i / w_max)^2 - b_i * (w_i / w_max) * v_i - c_i * v_i^2 = 0
    def cons3_rule(model, l):
        term1 = delta_p[l]
        term2 = a[l] * (w[l] / w_max)**2
        term3 = b[l] * (w[l] / w_max) * v[l]
        term4 = c[l] * v[l]**2
        return term1 - term2 - term3 - term4 == 0
    
    model.Constraint3 = pyo.Constraint(levels, rule=cons3_rule)

    # 4. v_i * N_p_i - x_i * V_tot = 0
    def cons4_rule(model, l):
        return v[l] * Np[l] == x[l] * v_tot

    model.Constraint4 = pyo.Constraint(levels, rule=cons4_rule)

    # 5. ΔP_tot * z_i - Δp_i * N_s_i = 0
    def cons5_rule(model, l):
        return delta_p_tot * z[l] == delta_p[l] * Ns[l]
    
    model.Constraint5 = pyo.Constraint(levels, rule=cons5_rule)

    # Constraints 6 - 11 are implemented in the bounds of the decision vars

    # 12. 0 <= P_i <= P_max_i
    def cons12_rule(model, l):
        return P[l] <= pmax[l]
    
    model.Constraint12 = pyo.Constraint(levels, rule=cons12_rule)

    # 14. P_i <= z_i * P_max_i
    def cons14_rule(model, l):
        return P[l] <= z[l] * pmax[l]
    
    model.Constraint14 = pyo.Constraint(levels, rule=cons14_rule)

    # 15. Δp_i <= z_i * ΔP_tot
    def cons15_rule(model, l):
        return delta_p[l] <= z[l] * delta_p_tot
    
    model.Constraint15 = pyo.Constraint(levels, rule=cons15_rule)

    # 16. v_i <= z_i * V_tot
    def cons16_rule(model, l):
        return v[l] <= z[l] * v_tot

    model.Constraint16 = pyo.Constraint(levels, rule=cons16_rule)

    # 17. w_i <= z_i * w_max
    def cons17_rule(model, l):
        return w[l] <= z[l] * w_max
    
    model.Constraint17 = pyo.Constraint(levels, rule=cons17_rule)

    # 18. N_p_i <= z_i * NP_max
    def cons18_rule(model, l):
        return Np[l] <= z[l] * Np_max
    
    model.Constraint18 = pyo.Constraint(levels, rule=cons18_rule)

    # 19. N_s_i <= z_i * NS_max
    def cons19_rule(model, l):
        return Ns[l] <= z[l] * Ns_max
    
    model.Constraint19 = pyo.Constraint(levels, rule=cons19_rule)

    # 20. x_i <= z_i
    def cons20_rule(model, l):
        return x[l] <= z[l]
    
    model.Constraint20 = pyo.Constraint(levels, rule=cons20_rule)

    # 21. N_p_i >= z_i
    def cons21_rule(model, l):
        return Np[l] >= z[l]
    
    model.Constraint21 = pyo.Constraint(levels, rule=cons21_rule)

    # 22. N_s_i >= z_i
    def cons22_rule(model, l):
        return Ns[l] >= z[l]
    
    model.Constraint22 = pyo.Constraint(levels, rule=cons22_rule)

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
        if solver_name == "scip":
            solver.options["limits/time"] = 300
            solver.options["limits/gap"] = 1e-9
            solver.options["display/verblevel"] = 4
        if solver is not None and solver.available(False):
            return solver_name, solver
    except Exception:
        pass
    return None, None


def create_neos_manager():
    """Create NEOS Solver Manager"""
    solver_manager = pyo.SolverManagerFactory("neos")
    return solver_manager


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
    print("=" * 70)
    for l in model.levels:
        print(f"Level {l} z: {model.z[l]()}")
        print(f"Level {l} Np: {model.Np[l]()}")
        print(f"Level {l} Ns: {model.Ns[l]()}")
        print(f"Level {l} x: {model.x[l]()}")
        print(f"Level {l} w: {model.w[l]()}")
        print(f"Level {l} v: {model.v[l]()}")
        print(f"Level {l} P: {model.P[l]()}")
        print(f"Level {l} Δp: {model.delta_p[l]()}")
        print("=" * 70)


if __name__ == "__main__":
    main()
