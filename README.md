# Stochastic and Robust Distribution Network Reconfiguration under PV and Load Uncertainty

Python/Pyomo/HiGHS framework that chooses the open/closed switches of a distribution feeder
(first-stage, scenario independent) so that expected (stochastic) or worst-case (robust)
operating cost is minimal under PV and load uncertainty.

Design references (ideas only, no code copied): `sometimesstudy/distribution-network-reconfiguration`
(DistFlow, directed spanning-tree / parent-arc radiality, loss minimisation) and
`Ahmadmohammadip/resilient-microgrid-optimization` (Pyomo + HiGHS MILP, LinDistFlow, explicit
radiality, scenarios).

## Install and run
```
pip install -r requirements.txt
python main_stochastic.py          # two-stage stochastic DNR, IEEE 33-bus demo (~5 s)
python main_robust.py              # two-stage robust DNR (C&CG), ~1 min
python -m pytest -q                # 21 tests
```
Useful options: `--radiality scf|directed`, `--solver`, `--time-limit`, `--max-switch-ops N`,
`--scenarios-csv FILE`, `--random-scenarios N`, `--network-excel FILE`,
`--branches-csv/--buses-csv/--meta-json`; robust: `--method ccg|box`, `--oracle heuristic|milp`,
`--gamma-load`, `--gamma-pv`, `--load-dev`, `--pv-dev`, `--pv-two-sided`, `--periods 12 19`.
Solver name and HiGHS options are configurable in `DNRConfig` (no commercial solver needed).

## Layout
`data/` network + scenario I/O and profiles - `models/` radiality, DistFlow block, stochastic and
robust models - `utils/` scenario generation, topology validation, AC power flow, results,
plots - `results/` outputs of the last run - `tests/`.

## Input formats (units: kW, kVAr, ohm, A, kV; internal p.u. on S_base, V_base)
* **Buses** (`data/example_buses.csv`): `bus_id, bus_type (root|load|pv|pv_load|ordinary), p_load_kw,
  q_load_kvar, v_min_pu, v_max_pu, pv_capacity_kw, load_profile, pv_profile`.
* **Branches** (`data/example_network.csv`): `branch_id, from_bus, to_bus, r_ohm, x_ohm, i_max_a,
  initial_status (1 closed), normally_open, switchable (0 = fixed), switch_cost ($/operation)`.
* **Meta** (`example_meta.json` or Excel sheet `meta`): `name, s_base_mva, v_base_kv, root_bus, v_root_pu`.
* **Excel**: workbook with sheets `buses`, `branches`, optional `meta` (`load_network_excel`).
* **Scenarios** (CSV or Excel sheet `scenarios`, long format; see `data/example_scenarios.csv`):
  `scenario_id, probability, period, duration_h (optional), bus_id, p_load_kw, q_load_kvar, pv_kw`
  (`pv_kw` = available PV before curtailment). Probabilities are normalised on request.

**Replace the feeder:** provide your own bus/branch tables (all candidate branches incl. ties, with
`initial_status`), set the meta values, and run `main_*.py` with the file options. The candidate
graph must connect all buses to the root. Time series: add one scenario per (hour, case) or use
`utils.scenario_generation`.

## Formulation
Per unit; branch l=(i->j); `s_l in {0,1}` (1 = closed) is first stage. Scenario k has load
`p_i^k, q_i^k` and available PV `pv_i^k`.

**LinDistFlow (second stage, per scenario)** - losses neglected in balance and drop:
* balance (non-root i): `sum_{l:to=i} P_l - sum_{l:from=i} P_l = p_i - shed_i - (pv_i - curt_i)`;
  `... Q ... = q_i - tan(phi_i) shed_i` (PV at unity power factor)
* drop: `|v2_i - v2_j - 2(r_l P_l + x_l Q_l)| <= M_l (1 - s_l)`, `v_min^2 <= v2 <= v_max^2`, `v2_root = V0^2`
* limits: `|P|,|Q| <= a s_l`, `|P|+|Q| <= sqrt(2) a s_l` with `a = S_max cos(pi/8)` (inscribed octagon,
  `S_max = sqrt(3) V_base I_max` at 1 p.u.)
* current/loss: `P2 >= 2cP - c^2`, `Q2 >= 2cQ - c^2` over breakpoints c (tangent lower envelope);
  `I2 = P2 + Q2`, `loss = r (P2 + Q2)`; reported loss therefore *under*-estimates `r(P^2+Q^2)`
  (post-solve, the exact quadratic and the AC loss are reported).
* `0 <= curt <= pv`, `0 <= shed <= p`.

**Radiality** (`models/radiality.py`): `scf` = commodity flow from root (N-1 units out, 1 unit
consumed per bus, flow only on closed branches) + `sum s = N-1`; `directed` = parent arcs
(one parent per non-root bus, none for root) + the same flow. Both forbid islands *and* loops; the
edge-count alone is never used. Tests compare the MILP to NetworkX minimum spanning trees.

**Stochastic**: `min sum_l c_l|s_l - s0_l| + sum_k pi_k Cost_k`,
`Cost_k = h_k [ S_base (c_loss Loss_k + c_curt Curt_k + c_shed Shed_k) + c_viol Viol_k ]`
(defaults 60 / 120 / 5000 $/MWh, h = 24 h). Optional: `max_switch_ops`, `exclusive_close_pairs`,
`soft_voltage`, fixed (non-switchable) branches.

**Robust** (`models/robust_dnr.py`): for each forecast period t,
`p_i = p0_i (1 + dev_L z_i)`, `z_i in [-1,1]`, `sum|z_i| <= Gamma_L`;
`pv_i = pv0_i - dn_i a_i (+ up_i b_i)`, `a_i,b_i in [0,1]`, `sum(a+b) <= Gamma_PV` (upward PV
deviation only with `--pv-two-sided`). Objective: switching cost + sum_t w_t * worst-case cost.
Solved by column-and-constraint generation (master = switches + one copy of the recourse
problem per generated worst case).
* `--oracle milp`: exact separation (dualised recourse LP, vertices of U as binaries, big-M
  linearisation of dual x binary; integer budgets). Verified on small feeders only: on the
  33-bus feeder this MILP did not solve in reasonable time.
* `--oracle heuristic` (default): single-deviation screening + 1-swap local search. The master is
  guaranteed to protect against every realisation found, but **this is not a proof of worst-case
  optimality**; results are labelled `exact=false`. The "upper bound" is the cost against worst
  cases found, not a certified bound.
* `--method box`: conservative extreme-realisation approximation, ignores the budget; not exact.

## Outputs (`results/`)
`*_solution.json` (switches, opened/closed lists, objective, per-scenario loss / curtailment /
shedding / voltages / flows / currents, solver status and time, topology and validation report),
`*_switches.csv`, `*_scenario_summary.csv`, `*_bus_voltages.csv`, `*_branch_flows.csv`,
`*_validation_report.json`, and figures (voltage profile, line loading, PV curtailment, load
shedding, final topology, C&CG convergence).

## Validation performed automatically
solver status, connectivity, no cycles, N-1 branches, tree structure, power-balance residual,
voltage and line limits, curtailment/shedding bounds, no flow on open branches (single switch plan
for all scenarios), plus a nonlinear AC backward/forward sweep (voltage, loss, current error and
limit checks; shown as warnings because LinDistFlow is an approximation). Infeasibility: see
`diagnose_infeasibility` (first-stage check, then a relaxed model showing which scenarios need
shedding/curtailment).

## Data sources and assumptions
IEEE 33-bus impedances/loads: Baran & Wu, IEEE Trans. Power Delivery 4(2), 1989 (12.66 kV, 10 MVA).
**Assumed by this project (not in the paper):** ampacities (400 A trunk / 250 A others), PV at
buses 18/25/33 (500/800/600 kW), substation 1.03 p.u., switching cost $5, load/PV profiles,
cost coefficients. With the paper's 1.0 p.u. substation voltage and a 0.95 p.u. limit, evening
scenarios would require load shedding.

## Limitations
LinDistFlow ignores losses in flows/voltages (AC check reports the error); all buses must be
energised (no islanding); constant power-factor load shedding; PV at unity power factor; scenarios
are independent operating points (no inter-temporal coupling); robust budgets are integers
(floored) and soft voltage is not supported in the robust model; the default robust separation is
heuristic (see above).
