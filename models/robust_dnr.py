"""Two-stage robust DNR with budgeted (cardinality-constrained) uncertainty.

    min_{s}  switching_cost(s) + sum_t w_t * max_{u in U_t}  min_{y in Y(s,u)}  Cost_t(y)

For every forecast period t (weight w_t):
  load   : p_i = p0_i (1 + dev_L z_i),  z_i = zp_i - zm_i in [-1,1],  sum_i |z_i| <= Gamma_L
           (q_i follows p_i at constant power factor)
  PV     : pv_i = pv0_i - dn_i*a_i + up_i*b_i,  a_i in [0,1] (shortfall),  b_i in [0,1] (surplus,
           only if ``pv_two_sided``),  sum_i (a_i + b_i) <= Gamma_PV.
  dn_i = pv_dev*pv0_i,  up_i = min(pv_dev*pv0_i, capacity_i - pv0_i).

Methods
-------
``solve_robust_ccg``  EXACT (finite-convergent) column-and-constraint generation.
    Master: first-stage s + one second-stage copy per generated worst-case realisation.
    Oracle: the recourse problem is an LP for fixed s with the uncertain data entering only through
    right-hand sides; it is dualised and max_u of the dual objective is a MILP once the
    realisation is restricted to the vertices of U (binary zp, zm, a, b; valid for INTEGER budgets
    because the worst case of a convex piecewise-linear function over a polytope is attained at a
    vertex) and the bilinear terms (dual x binary) are linearised with big-M.  Caveat: the big-M
    needs bounds on the dual variables of the uncertain rows; they are set from the cost scale and
    verified numerically (the oracle value must equal the true recourse cost at the returned
    realisation, else the bound is enlarged).
``solve_robust_box``   CONSERVATIVE APPROXIMATION: only two extreme realisations per period
    (all loads up & PV down; all loads down & PV up if two-sided), ignoring the budget.  Not exact
    for the budgeted set (it may be over- or under-protective on cost) -- use only as a fast heuristic.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

from data.load_scenarios import Scenario
from models.distflow import DNRConfig, build_operation_block
from models.stochastic_dnr import (build_first_stage, evaluate_topology, first_stage_cost,
                                   scenario_hours, scenario_to_pu, solve_model, switch_values)


@dataclass
class UncertaintySet:
    """Budgeted uncertainty set (fractions of forecast; budgets are integers)."""
    load_dev: float = 0.15
    pv_dev: float = 0.30
    gamma_load: int = 8
    gamma_pv: int = 2
    pv_two_sided: bool = False
    kind: str = "budget"            # "budget" or "box" (box: budgets = number of uncertain items)

    def budgets(self, n_load: int, n_pv: int) -> Tuple[int, int]:
        """Effective integer budgets (floored; box => all items)."""
        if self.kind == "box":
            return n_load, n_pv
        return min(int(math.floor(self.gamma_load)), n_load), min(int(math.floor(self.gamma_pv)), n_pv)


def uncertain_items(net, forecast: Scenario) -> Tuple[List[int], List[int]]:
    """Buses with uncertain load and uncertain PV at this forecast point."""
    loads = [b for b in net.load_buses if forecast.p_load_kw.get(b, 0.0) > 1e-9]
    pvs = [b for b in net.pv_buses if forecast.pv_kw.get(b, 0.0) > 1e-9]
    return loads, pvs


def pv_deviation_kw(net, forecast: Scenario, uset: UncertaintySet, b: int) -> Tuple[float, float]:
    """(downward, upward) PV deviation magnitude [kW] at bus b."""
    pv0 = forecast.pv_kw.get(b, 0.0)
    cap = net.pu_to_kw(net.pv_cap_pu[b])
    dn = uset.pv_dev * pv0
    up = min(uset.pv_dev * pv0, max(cap - pv0, 0.0)) if uset.pv_two_sided else 0.0
    return dn, up


def make_realization(net, forecast: Scenario, uset: UncertaintySet, zp=(), zm=(), pdn=(), pup=(),
                     sid: Optional[str] = None) -> Scenario:
    """Realisation of the uncertainty: sets of buses at the +/- load deviation / PV shortfall / surplus."""
    zp, zm, pdn, pup = set(zp), set(zm), set(pdn), set(pup)
    p, q = dict(forecast.p_load_kw), dict(forecast.q_load_kvar)
    for b in zp | zm:
        f = 1.0 + uset.load_dev * (1.0 if b in zp else -1.0)
        p[b] = forecast.p_load_kw[b] * f
        q[b] = forecast.q_load_kvar[b] * f
    pv = dict(forecast.pv_kw)
    for b in pdn | pup:
        dn, up = pv_deviation_kw(net, forecast, uset, b)
        pv[b] = pv[b] - (dn if b in pdn else 0.0) + (up if b in pup else 0.0)
    key = (tuple(sorted(zp)), tuple(sorted(zm)), tuple(sorted(pdn)), tuple(sorted(pup)))
    return Scenario(scenario_id=sid or f"{forecast.scenario_id}_wc", probability=forecast.probability,
                    p_load_kw=p, q_load_kvar=q, pv_kw=pv, period=forecast.period, duration_h=forecast.duration_h,
                    meta=dict(key=key))


def greedy_adverse(net, forecast: Scenario, uset: UncertaintySet) -> Scenario:
    """Heuristic adverse realisation inside U: the Gamma_L largest loads up, the Gamma_PV largest PV down."""
    loads, pvs = uncertain_items(net, forecast)
    gl, gp = uset.budgets(len(loads), len(pvs))
    zp = sorted(loads, key=lambda b: -forecast.p_load_kw[b])[:gl]
    pdn = sorted(pvs, key=lambda b: -forecast.pv_kw[b])[:gp]
    return make_realization(net, forecast, uset, zp=zp, pdn=pdn, sid=f"{forecast.scenario_id}_greedy")


class RobustMaster:
    """C&CG master: first-stage switches + one operation block per generated realisation."""

    def __init__(self, net, forecasts: Sequence[Scenario], cfg: DNRConfig):
        self.net, self.forecasts, self.cfg = net, list(forecasts), cfg
        m = self.m = pyo.ConcreteModel("robust_master")
        build_first_stage(m, net, cfg)
        m.T = pyo.RangeSet(0, len(forecasts) - 1)
        m.eta = pyo.Var(m.T, bounds=(0, None))       # worst-case operating cost of period t [$]
        m.cuts = pyo.ConstraintList()
        m.fs = pyo.Expression(expr=first_stage_cost(net, m.s, cfg))
        m.obj = pyo.Objective(expr=m.fs + sum(f.probability * m.eta[t] for t, f in enumerate(forecasts)))
        self.n_blocks = 0

    def add(self, t: int, real: Scenario) -> None:
        """Add a realisation (scenario columns + constraint eta_t >= cost)."""
        blk = pyo.Block()
        self.m.add_component(f"real_{self.n_blocks}", blk)
        self.n_blocks += 1
        p, q, tphi, pv = scenario_to_pu(self.net, real)
        build_operation_block(blk, self.net, self.m.s, p, q, tphi, pv, self.cfg, scenario_hours(real, self.cfg))
        self.m.cuts.add(self.m.eta[t] >= blk.cost)

    def solve(self) -> Dict[str, Any]:
        """Solve the master; returns solve info plus ``switches``."""
        info = solve_model(self.m, self.cfg)
        if not info["solution_loaded"]:
            raise RuntimeError(f"Robust master failed: {info['termination']}")
        info["switches"] = switch_values(self.m, self.net)
        return info


# ----------------------------------------------------------------------------- oracle
def _build_inner_lp(net, switches, forecast: Scenario, uset: UncertaintySet, cfg: DNRConfig):
    """Recourse LP for fixed topology with uncertainty entering through 'xi == value' rows."""
    m = pyo.ConcreteModel("inner")
    loads, pvs = uncertain_items(net, forecast)
    p0, q0, tphi, pv0 = scenario_to_pu(net, forecast)
    m.xzp = pyo.Var(loads)
    m.xzm = pyo.Var(loads)
    m.xdn = pyo.Var(pvs)
    m.xup = pyo.Var(pvs) if uset.pv_two_sided else None
    rowkey: Dict[int, Tuple[str, int]] = {}
    m.fix = pyo.ConstraintList()
    dev_pu: Dict[Tuple[str, int], float] = {}

    def fixrow(var, kind, i, dev):
        m.fix.add(var == 0.0)
        rowkey[id(m.fix[len(m.fix)])] = (kind, i)
        dev_pu[(kind, i)] = dev

    p_load, q_load, pv = dict(p0), dict(q0), dict(pv0)
    for i in loads:
        d = uset.load_dev * p0[i]
        fixrow(m.xzp[i], "zp", i, d * (1 + tphi[i]))
        fixrow(m.xzm[i], "zm", i, d * (1 + tphi[i]))
        p_load[i] = p0[i] + d * (m.xzp[i] - m.xzm[i])
        q_load[i] = tphi[i] * p_load[i]
    for i in pvs:
        dn, up = pv_deviation_kw(net, forecast, uset, i)
        dn, up = net.kw_to_pu(dn), net.kw_to_pu(up)
        fixrow(m.xdn[i], "pdn", i, dn)
        expr = pv0[i] - dn * m.xdn[i]
        if uset.pv_two_sided:
            fixrow(m.xup[i], "pup", i, up)
            expr = expr + up * m.xup[i]
        pv[i] = expr
    m.op = pyo.Block()
    build_operation_block(m.op, net, {l: float(switches[l]) for l in net.branch_ids},
                          p_load, q_load, tphi, pv, cfg, scenario_hours(forecast, cfg))
    m.obj = pyo.Objective(expr=m.op.cost)
    return m, rowkey, dev_pu, loads, pvs


def _extract_lp(m: pyo.ConcreteModel):
    """Rows/cols of a linear Pyomo model: rows=(con_id, {col: a}, sense, rhs); objective (c, const)."""
    cols: Dict[int, int] = {}
    vlist: List[Any] = []

    def idx(v):
        if id(v) not in cols:
            cols[id(v)] = len(vlist)
            vlist.append(v)
        return cols[id(v)]

    rows = []
    for con in m.component_data_objects(pyo.Constraint, active=True):
        rep = generate_standard_repn(con.body, compute_values=True)
        if not rep.is_linear():
            raise ValueError("non-linear constraint in inner LP")
        coef = {}
        for v, c in zip(rep.linear_vars, rep.linear_coefs):
            coef[idx(v)] = coef.get(idx(v), 0.0) + float(c)
        coef = {j: a for j, a in coef.items() if abs(a) > 0}
        if not coef:
            continue
        lo = None if con.lower is None else float(pyo.value(con.lower))
        up = None if con.upper is None else float(pyo.value(con.upper))
        const = float(rep.constant)
        if lo is not None and up is not None and abs(lo - up) < 1e-14:
            rows.append((id(con), coef, "==", lo - const))
        else:
            if lo is not None:
                rows.append((id(con), coef, ">=", lo - const))
            if up is not None:
                rows.append((id(con), coef, "<=", up - const))
    orep = generate_standard_repn(next(m.component_data_objects(pyo.Objective)).expr, compute_values=True)
    c = {}
    for v, a in zip(orep.linear_vars, orep.linear_coefs):
        c[idx(v)] = c.get(idx(v), 0.0) + float(a)
    for j, v in enumerate(vlist):          # variable bounds as rows
        if v.lb is not None:
            rows.append((None, {j: 1.0}, ">=", float(v.lb)))
        if v.ub is not None:
            rows.append((None, {j: 1.0}, "<=", float(v.ub)))
    return rows, len(vlist), c, float(orep.constant)


def solve_worst_case_oracle(net, switches, forecast: Scenario, uset: UncertaintySet, cfg: DNRConfig,
                            dual_factor: float = 4.0, time_limit: float = 120.0) -> Dict[str, Any]:
    """Worst-case realisation for fixed switches (dual MILP). Returns dict(real, value, info)."""
    inner, rowkey, dev_pu, loads, pvs = _build_inner_lp(net, switches, forecast, uset, cfg)
    rows, ncol, cvec, cconst = _extract_lp(inner)
    hours = scenario_hours(forecast, cfg)
    # Marginal value of uncertain data is at most the shedding price (only when shedding binds);
    # start from the curtailment/loss price scale and enlarge via dual_factor if the check fails.
    cmax = hours * net.s_base_mva * max(cfg.curtailment_cost_per_mwh, cfg.loss_cost_per_mwh)
    gl, gp = uset.budgets(len(loads), len(pvs))

    d = pyo.ConcreteModel("oracle")
    d.R = pyo.RangeSet(0, len(rows) - 1)
    bnd = {}
    d.y = pyo.Var(d.R)
    unc_rows = {}
    for r, (cid, coef, sense, rhs) in enumerate(rows):
        if cid in rowkey:
            unc_rows[r] = rowkey[cid]
            Y = max(dual_factor * cmax * dev_pu[rowkey[cid]], 1e-6 * cmax)
            d.y[r].setlb(-Y)
            d.y[r].setub(Y)
        elif sense == ">=":
            d.y[r].setlb(0.0)
        elif sense == "<=":
            d.y[r].setub(0.0)
    colrows: Dict[int, List[Tuple[int, float]]] = {j: [] for j in range(ncol)}
    for r, (cid, coef, sense, rhs) in enumerate(rows):
        for j, a in coef.items():
            colrows[j].append((r, a))
    d.dcon = pyo.ConstraintList()
    for j in range(ncol):
        if colrows[j]:
            d.dcon.add(pyo.quicksum(a * d.y[r] for r, a in colrows[j]) == cvec.get(j, 0.0))
    # uncertainty binaries
    d.zp = pyo.Var(loads, domain=pyo.Binary)
    d.zm = pyo.Var(loads, domain=pyo.Binary)
    d.pdn = pyo.Var(pvs, domain=pyo.Binary)
    d.pup = pyo.Var(pvs, domain=pyo.Binary) if uset.pv_two_sided else None
    d.unc = pyo.ConstraintList()
    for i in loads:
        d.unc.add(d.zp[i] + d.zm[i] <= 1)
    if loads:
        d.unc.add(sum(d.zp[i] + d.zm[i] for i in loads) <= gl)
    if pvs:
        pv_terms = sum(d.pdn[i] for i in pvs) + (sum(d.pup[i] for i in pvs) if uset.pv_two_sided else 0)
        d.unc.add(pv_terms <= gp)
    if uset.pv_two_sided:
        for i in pvs:
            d.unc.add(d.pdn[i] + d.pup[i] <= 1)
    d.w = pyo.Var(list(unc_rows.keys()))
    d.lin = pyo.ConstraintList()
    objterms = []
    for r, (cid, coef, sense, rhs) in enumerate(rows):
        if r in unc_rows:
            kind, i = unc_rows[r]
            b = {"zp": d.zp, "zm": d.zm, "pdn": d.pdn, "pup": d.pup}[kind][i]
            Y = d.y[r].ub
            d.lin.add(d.w[r] <= Y * b)
            d.lin.add(d.w[r] >= -Y * b)
            d.lin.add(d.w[r] <= d.y[r] + Y * (1 - b))
            d.lin.add(d.w[r] >= d.y[r] - Y * (1 - b))
            objterms.append(d.w[r])
        elif rhs != 0.0:
            objterms.append(rhs * d.y[r])
    d.obj = pyo.Objective(expr=pyo.quicksum(objterms) + cconst, sense=pyo.maximize)
    info = solve_model_max(d, cfg, time_limit)
    zp = [i for i in loads if pyo.value(d.zp[i]) > 0.5]
    zm = [i for i in loads if pyo.value(d.zm[i]) > 0.5]
    pdn = [i for i in pvs if pyo.value(d.pdn[i]) > 0.5]
    pup = [i for i in pvs if uset.pv_two_sided and pyo.value(d.pup[i]) > 0.5]
    real = make_realization(net, forecast, uset, zp, zm, pdn, pup)
    return dict(real=real, value=info["objective"], info=info)


def _batch_costs(net, switches, reals: Sequence[Scenario], cfg: DNRConfig) -> List[float]:
    """True recourse cost [$] of several realisations for a fixed topology (one batched LP)."""
    n = len(reals)
    tmp = [Scenario(scenario_id=f"r{k}", probability=1.0 / n, p_load_kw=r.p_load_kw, q_load_kvar=r.q_load_kvar,
                    pv_kw=r.pv_kw, period=r.period, duration_h=r.duration_h) for k, r in enumerate(reals)]
    m, info = evaluate_topology(net, switches, tmp, cfg)
    if not info["solution_loaded"]:
        raise RuntimeError(f"Recourse LP failed: {info['termination']}")
    return [float(pyo.value(m.ops[k].cost)) for k in range(n)]


def heuristic_worst_case(net, switches, forecast: Scenario, uset: UncertaintySet, cfg: DNRConfig,
                         n_swap_candidates: int = 6, max_rounds: int = 6) -> Dict[str, Any]:
    """Heuristic separation: first-order screening of single deviations + 1-swap local search.

    NOT a proof of worst-case optimality (use ``oracle='milp'`` for that on small feeders), but every
    realisation returned lies inside U and its cost is the TRUE recourse cost of a feasible point.
    """
    loads, pvs = uncertain_items(net, forecast)
    gl, gp = uset.budgets(len(loads), len(pvs))
    singles: List[Tuple[str, int, Scenario]] = []
    for b in loads:
        singles.append(("zp", b, make_realization(net, forecast, uset, zp=[b])))
        singles.append(("zm", b, make_realization(net, forecast, uset, zm=[b])))
    for b in pvs:
        singles.append(("pdn", b, make_realization(net, forecast, uset, pdn=[b])))
        if uset.pv_two_sided:
            singles.append(("pup", b, make_realization(net, forecast, uset, pup=[b])))
    costs = _batch_costs(net, switches, [make_realization(net, forecast, uset)] + [x[2] for x in singles], cfg)
    g0, gains = costs[0], costs[1:]
    # best direction per item and ranked candidate lists
    best_dir: Dict[Tuple[str, int], Tuple[str, float]] = {}
    for (kind, b, _), gv in zip(singles, gains):
        grp = ("L", b) if kind in ("zp", "zm") else ("P", b)
        if grp not in best_dir or gv > best_dir[grp][1]:
            best_dir[grp] = (kind, gv)
    rank_l = sorted([b for b in loads], key=lambda b: -best_dir[("L", b)][1])
    rank_p = sorted([b for b in pvs], key=lambda b: -best_dir[("P", b)][1])

    def build(sel_l, sel_p):
        sets = dict(zp=[], zm=[], pdn=[], pup=[])
        for b in sel_l:
            sets[best_dir[("L", b)][0]].append(b)
        for b in sel_p:
            sets[best_dir[("P", b)][0]].append(b)
        return make_realization(net, forecast, uset, **sets)

    sel_l, sel_p = rank_l[:gl], rank_p[:gp]
    cur = build(sel_l, sel_p)
    cur_cost = _batch_costs(net, switches, [cur], cfg)[0]
    for _ in range(max_rounds):
        cands = []
        for out_b in sel_l:
            for in_b in [b for b in rank_l if b not in sel_l][:n_swap_candidates]:
                cands.append(([in_b if x == out_b else x for x in sel_l], sel_p))
        for out_b in sel_p:
            for in_b in [b for b in rank_p if b not in sel_p][:n_swap_candidates]:
                cands.append((sel_l, [in_b if x == out_b else x for x in sel_p]))
        if not cands:
            break
        reals = [build(a, b) for a, b in cands]
        cs = _batch_costs(net, switches, reals, cfg)
        k = int(np.argmax(cs))
        if cs[k] > cur_cost + 1e-9:
            sel_l, sel_p = cands[k]
            cur, cur_cost = reals[k], cs[k]
        else:
            break
    return dict(real=cur, value=cur_cost, nominal_cost=g0)


def solve_model_max(model, cfg: DNRConfig, time_limit: float) -> Dict[str, Any]:
    """Solve a maximisation model (objective value reported as is)."""
    info = solve_model(model, cfg, time_limit=time_limit)
    if not info["solution_loaded"]:
        raise RuntimeError(f"Oracle failed: {info['termination']}")
    return info


def solve_robust_ccg(net, forecasts: Sequence[Scenario], uset: UncertaintySet, cfg: DNRConfig,
                     max_iter: int = 20, tol: float = 1e-3, oracle_time: float = 120.0,
                     verbose: bool = True, oracle: str = "heuristic") -> Dict[str, Any]:
    """Column-and-constraint generation.

    ``oracle="milp"``: exact separation by the dual MILP (practical only for small feeders; result is
    exact when it converges).  ``oracle="heuristic"`` (default): screening + local-search separation;
    the master is then guaranteed to protect against every generated realisation, but a worse
    realisation inside U may exist -- the reported "upper bound" is the cost against the worst
    realisations FOUND, not a certified bound.
    """
    t_start = time.perf_counter()
    master = RobustMaster(net, forecasts, cfg)
    seen: List[set] = [set() for _ in forecasts]
    for t, f in enumerate(forecasts):
        nominal = make_realization(net, f, uset, sid=f"{f.scenario_id}_nominal")
        g = greedy_adverse(net, f, uset)
        for r in (nominal, g):
            if r.meta["key"] not in seen[t]:
                master.add(t, r)
                seen[t].add(r.meta["key"])
    UB, LB, best = math.inf, -math.inf, None
    hist: List[Dict[str, Any]] = []
    dual_factor = 4.0
    converged = False
    for it in range(1, max_iter + 1):
        minfo = master.solve()
        sw = minfo["switches"]
        LB = max(LB, minfo["objective"])
        fs = float(pyo.value(master.m.fs))
        reals, oracle_vals = [], []
        for t, f in enumerate(forecasts):
            for attempt in range(4):
                if oracle == "heuristic":
                    o = heuristic_worst_case(net, sw, f, uset, cfg)
                    true_q = o["value"]
                    break
                o = solve_worst_case_oracle(net, sw, f, uset, cfg, dual_factor, oracle_time)
                m_eval, ev = evaluate_topology(net, sw, [o["real"]], cfg)
                true_q = float(pyo.value(m_eval.ops[0].cost))
                if o["value"] >= true_q - 1e-3 * max(1.0, abs(true_q)) or attempt == 3:
                    break
                dual_factor = min(dual_factor * 3.0, 4.0 * cfg.shedding_cost_per_mwh
                                  / max(cfg.curtailment_cost_per_mwh, cfg.loss_cost_per_mwh))  # enlarge
            reals.append((o["real"], true_q))
            oracle_vals.append(o["value"])
        total = fs + sum(f.probability * q for f, (_, q) in zip(forecasts, reals))
        if total < UB:
            UB, best = total, dict(switches=dict(sw), reals=[r for r, _ in reals], qs=[q for _, q in reals])
        gap = (UB - LB) / max(1.0, abs(UB))
        hist.append(dict(iteration=it, lower_bound=LB, upper_bound=UB, gap=gap,
                         master_objective=minfo["objective"], oracle_values=oracle_vals,
                         true_worst_costs=[q for _, q in reals], n_realizations=master.n_blocks,
                         closed=[l for l in net.branch_ids if sw[l] == 1]))
        if verbose:
            print(f"  C&CG iter {it:2d}: LB={LB:10.3f}  UB={UB:10.3f}  gap={gap:.2e}  "
                  f"worst costs={[round(q, 2) for _, q in reals]}")
        if gap <= tol:
            converged = True
            break
        added = 0
        for t, (r, _) in enumerate(reals):
            if r.meta["key"] not in seen[t]:
                master.add(t, r)
                seen[t].add(r.meta["key"])
                added += 1
        if added == 0:
            if verbose:
                print("  No new worst-case realisation found; stopping (bounds as reported).")
            break
    exact = (oracle == "milp") and converged
    return dict(method=("C&CG with exact MILP oracle" if oracle == "milp"
                        else "C&CG with heuristic separation (screening + 1-swap local search)"),
                exact=exact, oracle=oracle, converged=converged,
                switches=best["switches"], lower_bound=LB, upper_bound=UB,
                worst_case_total_cost=UB, worst_realizations=best["reals"], worst_costs=best["qs"],
                history=hist, dual_factor=dual_factor, time_s=time.perf_counter() - t_start,
                uncertainty_set=uset.__dict__)


def solve_robust_box(net, forecasts: Sequence[Scenario], uset: UncertaintySet, cfg: DNRConfig,
                     verbose: bool = True) -> Dict[str, Any]:
    """Conservative approximation: protect against the two box-extreme realisations per period."""
    t0 = time.perf_counter()
    master = RobustMaster(net, forecasts, cfg)
    reals = []
    for t, f in enumerate(forecasts):
        loads, pvs = uncertain_items(net, f)
        up = make_realization(net, f, uset, zp=loads, pdn=pvs, sid=f"{f.scenario_id}_box_adverse")
        master.add(t, up)
        reals.append(up)
        if uset.pv_two_sided:
            lo = make_realization(net, f, uset, zm=loads, pup=pvs, sid=f"{f.scenario_id}_box_benign")
            master.add(t, lo)
            reals.append(lo)
        master.add(t, make_realization(net, f, uset, sid=f"{f.scenario_id}_nominal"))
    info = master.solve()
    sw = info["switches"]
    return dict(method="Box-extreme approximation (conservative, NOT exact)", exact=False, converged=True,
                switches=sw, lower_bound=None, upper_bound=info["objective"],
                worst_case_total_cost=info["objective"], worst_realizations=reals, history=[],
                time_s=time.perf_counter() - t0, uncertainty_set=uset.__dict__)


def sample_uncertainty_set(net, forecast: Scenario, uset: UncertaintySet, n: int, seed: int = 7) -> List[Scenario]:
    """Random vertices of U (Gamma items at +-1) -- used to sanity-check the robust solution."""
    rng = np.random.default_rng(seed)
    loads, pvs = uncertain_items(net, forecast)
    gl, gp = uset.budgets(len(loads), len(pvs))
    out = []
    for k in range(n):
        idx = rng.choice(len(loads), size=gl, replace=False) if gl else []
        zp, zm = [], []
        for i in idx:
            (zp if rng.random() < 0.5 else zm).append(loads[int(i)])
        pj = rng.choice(len(pvs), size=gp, replace=False) if gp else []
        pdn, pup = [], []
        for i in pj:
            (pup if (uset.pv_two_sided and rng.random() < 0.5) else pdn).append(pvs[int(i)])
        out.append(make_realization(net, forecast, uset, zp, zm, pdn, pup, sid=f"{forecast.scenario_id}_smp{k}"))
    return out
