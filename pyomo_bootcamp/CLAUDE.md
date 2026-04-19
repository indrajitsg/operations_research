# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pyomo-based solutions for problems from the Udemy course "The Complete Pyomo Bootcamp A-Z". Each top-level numbered script (`01_*.py` through `11_*.py`) is a self-contained optimization problem spanning LP → MIP → NLP → MINLP. The `test_*.py` files verify individual solver installations. `sample1.py` / `sample2.py` are reference templates.

## Environment and commands

The project uses `uv` (see `uv.lock`, `pyproject.toml`). Python ≥ 3.11.

- Install deps: `uv sync`
- Run a problem: `uv run python 01_leather_lp.py [solver]`
- Solver installation smoke tests: `uv run python test_ipopt.py`, `test_scip.py`, `test_knitro.py`

There is no lint/test/build tooling configured — these are standalone scripts.

## Script conventions

Every numbered script follows the same skeleton, so stick to it when adding or editing models:

1. `build_model()` — builds and returns a `pyo.ConcreteModel` (Sets → Params → Vars → Objective → Constraints, in that order, with local aliases like `x = model.x` to keep rule bodies readable).
2. `choose_solver(solver_name)` — wraps `SolverFactory`, returns `(name, solver)` or `(None, None)`. Candidate solver list in the docstring: `gurobi`, `cplex_direct`, `knitroampl`, `baron`, `highs`, `scip`, `ipopt`. Knitro scripts set `par_numthreads=2`; `test_knitro.py` additionally enables multistart (`ms_enable=1`).
3. `main()` — reads `sys.argv[1]` as the solver name, **defaults to `highs`** when omitted, solves with `tee=True`, then prints objective and variable values under a `print_banner("RESULTS")` header.

Keep this shape when adding a new problem file. Do not introduce argparse, config files, or a shared utility module — duplication across files is intentional (the scripts are teaching material).

## NEOS-backed solvers

`test_neos.py` and `11_coil_design_minlp.py` submit jobs to the NEOS server for solvers not installed locally (e.g. `baron`, `bonmin`, `couenne`, `knitro`, `minlp`). Pattern:

```python
os.environ['NEOS_EMAIL'] = '...'
solver_manager = pyo.SolverManagerFactory("neos")
action_handle = solver_manager.queue(model, opt=solver_name)
results = solver_manager.wait_for(action_handle)
```

`11_coil_design_minlp.py` dispatches to NEOS when invoked as `python 11_coil_design_minlp.py neos <neos_solver_name>` — otherwise it uses the local solver path. When extending, preserve this dual-path dispatch.

## Problem-specific notes

- `06_travelling_salesman.py` uses MTZ subtour elimination: the auxiliary `u[i]` rank variables and the `u[i] - u[j] + N*x[i,j] <= N - 1` constraint over non-depot pairs are load-bearing — do not drop them.
- `11_coil_design_minlp.py` reads wire-gauge data from `S7P2_Data.csv` (path is relative, so run from the repo root) and includes a `print_post_solution_audit()` that re-derives every constrained quantity and flags violations. Update the audit alongside any constraint change.
- `scip.set` is a SCIP parameter file (very large); pass via `solver.options` or SCIP's `-s` flag rather than editing it casually.
