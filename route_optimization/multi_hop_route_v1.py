"""
Multi-hop route planner on a schedules SQLite DB.

Requirements:
  - Python 3.x
  - sqlite3 (stdlib)
  - gurobipy (optional; for --solver gurobi)
  - pyomo (optional; for --solver pyomo)

Table used:
  schedules(depicao TEXT, arricao TEXT, distance REAL, ...)

Usage examples:
  python multi_hop_route.py --db mydb.sqlite --src KSFO --dst KJFK --min 200 --max 500 --solver gurobi
  python multi_hop_route.py --db mydb.sqlite --src EGLL --dst LFPG --min 150 --max 600 --solver pyomo
  python multi_hop_route.py --db mydb.sqlite --src KLAX --dst KJFK --min 200 --max 500 --solver dijkstra
"""

import argparse
import sqlite3
import sys
from collections import defaultdict, deque
import math
import heapq


def fetch_edges(db_path, dmin, dmax):
    """
    Read legs from SQLite and filter by distance bounds.
    Groups duplicates and keeps the minimum distance per (depicao, arricao).
    Returns list of arcs (u, v, dist).
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # Inclusive bounds by default; change to > and < if you need strict
    q = """
        SELECT TRIM(UPPER(depicao)) AS dep,
               TRIM(UPPER(arricao)) AS arr,
               MIN(CAST(distance AS REAL)) AS dist
        FROM schedules
        WHERE depicao IS NOT NULL AND arricao IS NOT NULL
          AND TRIM(depicao) <> '' AND TRIM(arricao) <> ''
          AND depicao <> arricao
          AND CAST(distance AS REAL) >= ? AND CAST(distance AS REAL) <= ?
        GROUP BY dep, arr
    """
    cur.execute(q, (dmin, dmax))
    rows = cur.fetchall()
    con.close()
    edges = [(dep, arr, float(dist)) for (dep, arr, dist) in rows]
    return edges


def build_graph(edges):
    """
    From list of (u, v, w) build:
      - nodes (set)
      - arcs (list of (u,v))
      - cost dict {(u,v): w}
      - OUT and IN adjacency dicts mapping node -> list of (u,v)
    """
    nodes = set()
    arcs = []
    cost = {}
    OUT = defaultdict(list)
    IN = defaultdict(list)

    for u, v, w in edges:
        nodes.add(u); nodes.add(v)
        arcs.append((u, v))
        cost[(u, v)] = float(w)
        OUT[u].append((u, v))
        IN[v].append((u, v))

    # Ensure every node exists in OUT/IN even if empty
    for n in list(nodes):
        OUT.setdefault(n, [])
        IN.setdefault(n, [])
    return nodes, arcs, cost, OUT, IN


def prune_to_reachable(nodes, arcs, cost, OUT, IN, src, dst):
    """
    Reduce problem size: keep only nodes/edges that are on some path from src to dst.
    Returns (nodes2, arcs2, cost2, OUT2, IN2).
    """
    # Forward reachability from src
    f_reach = set()
    stack = [src]
    while stack:
        u = stack.pop()
        if u in f_reach: 
            continue
        f_reach.add(u)
        for (uu, vv) in OUT.get(u, []):
            if vv not in f_reach:
                stack.append(vv)

    # Reverse reachability to dst (i.e., from nodes that can reach dst)
    r_reach = set()
    stack = [dst]
    while stack:
        v = stack.pop()
        if v in r_reach:
            continue
        r_reach.add(v)
        for (uu, vv) in IN.get(v, []):
            if uu not in r_reach:
                stack.append(uu)

    keep_nodes = f_reach & r_reach
    if not keep_nodes:
        # Keep at least src and dst so infeasibility is clear later
        keep_nodes = {src, dst}

    arcs2 = [(i, j) for (i, j) in arcs if (i in keep_nodes and j in keep_nodes)]
    cost2 = {a: cost[a] for a in arcs2}

    OUT2 = defaultdict(list)
    IN2 = defaultdict(list)
    for (i, j) in arcs2:
        OUT2[i].append((i, j))
        IN2[j].append((i, j))
    for n in keep_nodes:
        OUT2.setdefault(n, [])
        IN2.setdefault(n, [])

    return keep_nodes, arcs2, cost2, OUT2, IN2


def reconstruct_path_from_selected(selected_set, cost, src, dst):
    """
    selected_set: set of arcs (i,j) chosen by the solver (x[i,j] ~ 1).
    Returns (route_nodes_list, total_distance). Raises ValueError on discontinuity.
    """
    succ = {}
    for (i, j) in selected_set:
        if i in succ:
            # Shouldn't happen with degree<=1 constraints, but guard anyway
            raise ValueError(f"Multiple outgoing arcs from {i}.")
        succ[i] = j

    path = [src]
    seen = {src}
    cur = src
    total = 0.0

    while cur != dst:
        if cur not in succ:
            raise ValueError("No continuous path (stuck before reaching destination).")
        nxt = succ[cur]
        total += cost[(cur, nxt)]
        if nxt in seen:
            raise ValueError("Cycle detected during path reconstruction.")
        path.append(nxt)
        seen.add(nxt)
        cur = nxt

    return path, total


# ---------- Dijkstra (fallback) ----------
def dijkstra_shortest_path(OUT, cost, src, dst):
    """
    Dijkstra on directed graph (positive weights).
    Returns (path_nodes, total_distance) or (None, None) if unreachable.
    """
    dist = {}
    prev = {}
    for n in OUT.keys():
        dist[n] = math.inf
        prev[n] = None
    if src not in dist or dst not in dist:
        return None, None

    dist[src] = 0.0
    h = [(0.0, src)]

    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for (uu, vv) in OUT[u]:
            alt = d + cost[(uu, vv)]
            if alt < dist[vv]:
                dist[vv] = alt
                prev[vv] = u
                heapq.heappush(h, (alt, vv))

    if dist[dst] == math.inf:
        return None, None

    # Reconstruct
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[dst]


# ---------- Gurobi (native) ----------
def solve_with_gurobi(nodes, arcs, cost, OUT, IN, src, dst, max_hops=None, timelimit=None, mipgap=None):
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError("gurobipy not available.") from e

    m = gp.Model("st_min_distance")
    m.Params.OutputFlag = 1
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if mipgap is not None:
        m.Params.MIPGap = float(mipgap)

    # Variables: x[i,j] in {0,1}
    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")

    # Objective: minimize sum distance * x
    m.setObjective(gp.quicksum(cost[a] * x[a] for a in arcs), GRB.MINIMIZE)

    # Flow conservation (single unit from src to dst)
    for n in nodes:
        out_expr = gp.quicksum(x[a] for a in OUT.get(n, []))
        in_expr = gp.quicksum(x[a] for a in IN.get(n, []))
        rhs = 1 if n == src else (-1 if n == dst else 0)
        m.addConstr(out_expr - in_expr == rhs, name=f"flow_{n}")

        # Degree limits (avoid branching). Not strictly necessary but keeps solution clean.
        m.addConstr(out_expr <= 1, name=f"deg_out_{n}")
        m.addConstr(in_expr  <= 1, name=f"deg_in_{n}")

    if max_hops is not None:
        m.addConstr(gp.quicksum(x[a] for a in arcs) <= int(max_hops), name="max_hops")

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi ended with status {m.Status}.")

    # Extract selected arcs
    chosen = {(i, j) for (i, j) in arcs if x[(i, j)].X > 0.5}
    if not chosen:
        return None, None

    try:
        path, total = reconstruct_path_from_selected(chosen, cost, src, dst)
    except ValueError:
        # As a robust fallback, attempt to run Dijkstra on the subgraph induced by chosen arcs
        # or simply declare infeasible.
        raise

    return path, total


# ---------- Pyomo (+Gurobi) ----------
def solve_with_pyomo(nodes, arcs, cost, OUT, IN, src, dst, max_hops=None, timelimit=None, mipgap=None):
    import pyomo.environ as pyo

    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=list(nodes))
    model.A = pyo.Set(within=model.N * model.N, initialize=arcs, dimen=2)
    model.c = pyo.Param(model.A, initialize=lambda m, i, j: float(cost[(i, j)]), within=pyo.Reals)

    model.x = pyo.Var(model.A, within=pyo.Binary)

    # Convenience: store OUT/IN as Pyomo indexed sets of arcs
    OUT_dict = {n: OUT.get(n, []) for n in nodes}
    IN_dict = {n: IN.get(n, []) for n in nodes}

    model.OUT = pyo.Set(model.N, dimen=2, initialize=OUT_dict)
    model.IN  = pyo.Set(model.N, dimen=2, initialize=IN_dict)

    def flow_rule(m, n):
        out_sum = sum(m.x[i, j] for (i, j) in m.OUT[n])
        in_sum  = sum(m.x[i, j] for (i, j) in m.IN[n])
        rhs = 1 if n == src else (-1 if n == dst else 0)
        return out_sum - in_sum == rhs

    model.flow = pyo.Constraint(model.N, rule=flow_rule)

    # Degree constraints (optional but helpful)
    model.deg_out = pyo.Constraint(model.N, rule=lambda m, n: sum(m.x[i, j] for (i, j) in m.OUT[n]) <= 1)
    model.deg_in  = pyo.Constraint(model.N, rule=lambda m, n: sum(m.x[i, j] for (i, j) in m.IN[n]) <= 1)

    if max_hops is not None:
        model.max_hops = pyo.Constraint(expr=sum(model.x[i, j] for (i, j) in model.A) <= int(max_hops))

    model.obj = pyo.Objective(expr=sum(model.c[i, j] * model.x[i, j] for (i, j) in model.A), sense=pyo.minimize)

    opt = pyo.SolverFactory("gurobi")
    if timelimit is not None:
        opt.options["TimeLimit"] = float(timelimit)
    if mipgap is not None:
        opt.options["MIPGap"] = float(mipgap)

    res = opt.solve(model, tee=False)

    # Extract solution
    chosen = {(i, j) for (i, j) in model.A if pyo.value(model.x[i, j]) > 0.5}
    if not chosen:
        return None, None

    path, total = reconstruct_path_from_selected(chosen, cost, src, dst)
    return path, total


def format_route(route_nodes, total_nm):
    legs = []
    for i in range(len(route_nodes) - 1):
        legs.append(f"{route_nodes[i]} → {route_nodes[i+1]}")
    route_str = "  ".join(legs)
    return route_str, total_nm


def main():
    ap = argparse.ArgumentParser(description="Multi-hop flight planner (min total distance with leg distance bounds).")
    ap.add_argument("--db", required=True, help="Path to SQLite database.")
    ap.add_argument("--src", required=True, help="Origin airport ICAO/IATA code (e.g., KSFO).")
    ap.add_argument("--dst", required=True, help="Destination airport ICAO/IATA code (e.g., KJFK).")
    ap.add_argument("--min", dest="dmin", type=float, default=200.0, help="Minimum leg distance (nm). Default 200.")
    ap.add_argument("--max", dest="dmax", type=float, default=500.0, help="Maximum leg distance (nm). Default 500.")
    ap.add_argument("--max-hops", dest="max_hops", type=int, default=None, help="Optional cap on number of legs.")
    ap.add_argument("--solver", choices=["gurobi", "pyomo", "dijkstra"], default="gurobi",
                    help="Optimization backend. Default: gurobi (native).")
    ap.add_argument("--timelimit", type=float, default=None, help="Optional solver time limit (seconds).")
    ap.add_argument("--mipgap", type=float, default=None, help="Optional MIP relative gap (e.g., 0.001).")
    args = ap.parse_args()

    src = args.src.strip().upper()
    dst = args.dst.strip().upper()
    dmin = float(args.dmin)
    dmax = float(args.dmax)

    if dmin <= 0 or dmax <= 0 or dmin > dmax:
        print("ERROR: Invalid distance bounds. Ensure 0 < min <= max.", file=sys.stderr)
        sys.exit(2)
    if src == dst:
        print("ERROR: Source and destination must differ.", file=sys.stderr)
        sys.exit(2)

    edges = fetch_edges(args.db, dmin, dmax)
    if not edges:
        print("No legs found within the specified distance range. Try widening bounds.", file=sys.stderr)
        sys.exit(1)

    nodes, arcs, cost, OUT, IN = build_graph(edges)
    if src not in nodes or dst not in nodes:
        # They might still be connected via other legs not meeting bounds; warn early
        # but continue to pruning which will likely result in infeasibility.
        pass

    nodes2, arcs2, cost2, OUT2, IN2 = prune_to_reachable(nodes, arcs, cost, OUT, IN, src, dst)
    if not arcs2:
        print("No feasible route (graph disconnected under given leg bounds).", file=sys.stderr)
        sys.exit(1)

    try:
        if args.solver == "gurobi":
            route, total = solve_with_gurobi(nodes2, arcs2, cost2, OUT2, IN2, src, dst,
                                             max_hops=args.max_hops, timelimit=args.timelimit, mipgap=args.mipgap)
        elif args.solver == "pyomo":
            route, total = solve_with_pyomo(nodes2, arcs2, cost2, OUT2, IN2, src, dst,
                                            max_hops=args.max_hops, timelimit=args.timelimit, mipgap=args.mipgap)
        else:  # dijkstra
            route, total = dijkstra_shortest_path(OUT2, cost2, src, dst)
    except Exception as e:
        print(f"Solver error: {e}", file=sys.stderr)
        sys.exit(1)

    if route is None:
        print("No feasible route found.", file=sys.stderr)
        sys.exit(1)

    route_str, total_nm = format_route(route, total)
    print("\n=== Optimal Route ===")
    print(route_str)
    print(f"Total distance: {total_nm:.1f} nm")
    print(f"Legs: {len(route)-1}")
    if args.max_hops:
        print(f"(max_hops constraint = {args.max_hops})")
    print()
    # Optionally list each leg with distance
    print("Leg breakdown:")
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        print(f"  {u} -> {v} : {cost2[(u, v)]:.1f} nm")


if __name__ == "__main__":
    main()
