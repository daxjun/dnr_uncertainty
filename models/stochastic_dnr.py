"""Two-stage scenario-based stochastic DNR (MILP).

First stage (here-and-now, scenario independent): switch status ``m.s[l]``, radiality variables.
Second stage (per scenario k, block ``m.ops[k]``): P, Q, I2 (=P2+Q2), v2, p_curt, p_shed.

    min  sum_l c_l |s_l - s0_l|  +  sum_k pi_k * Cost_k(s, y_k)

Cost_k = hours_k * [ S_base (c_loss*Loss_k + c_curt*Curt_k + c_shed*Shed_k) + c_viol*Viol_k ].
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pyomo.environ as pyo

from data.load_scenarios import Scenario, validate_scenarios
from models.distflow import DNRConfig, build_operation_block
from models.radiality import (add_radiality_constraints, add_switch_variables,
                              add_switching_limits, switching_cost_expr)


def scenario_to_pu(net, sc: Scenario):
    """Convert a Scenario (kW/kVAr) to per-unit dicts (p_load, q_load, tanphi, pv)."""
    p = {b: net.kw_to_pu(sc.p_load_kw.get(b, 0.0)) for b in net.bus_ids}
    q = {b: net.kw_to_pu(sc.q_load_kvar.get(b, 0.0)) for b in net.bus_ids}
    tphi = {b: (q[b] / p[b] if p[b] > 1e-12 else 0.0) for b in net.bus_ids}
    pv = {b: net.kw_to_pu(sc.pv_kw.get(b, 0.0)) for b in net.bus_ids}
    return p, q, tphi, pv


def scenario_hours(sc: Scenario, cfg: DNRConfig) -> float:
    """Duration [h] represented by a scenario."""
    return sc.duration_h if sc.duration_h is not None else cfg.default_hours


def build_first_stage(m: pyo.ConcreteModel, net, cfg: DNRConfig) -> None:
    """Switch variables, radiality and switching limits on model ``m``."""
    add_switch_variables(m, net)
    add_radiality_constraints(m, net, cfg.radiality_method)
    add_switching_limits(m, net, cfg.max_switch_ops, cfg.exclusive_close_pairs)


def first_stage_cost(net, s, cfg: DNRConfig):
    """Switching cost expression [$] (0 if disabled)."""
    return switching_cost_expr(net, s) if cfg.use_switching_cost else 0.0


def build_stochastic_model(net, scenarios: Sequence[Scenario], cfg: DNRConfig,
                           fixed_switches: Optional[Mapping[int, int]] = None) -> pyo.ConcreteModel:
    """Build the stochastic DNR model.

    If ``fixed_switches`` is given, topology is fixed (numbers) and the model is a pure LP
    (used for evaluating a configuration); otherwise ``m.s`` are binary first-stage variables.
    """
    validate_scenarios(net, scenarios)
    m = pyo.ConcreteModel("stochastic_dnr")
    if fixed_switches is None:
        build_first_stage(m, net, cfg)
        s = m.s
    else:
        s = {l: float(fixed_switches[l]) for l in net.branch_ids}
    m.scenarios = list(scenarios)
    m.ops: List[pyo.Block] = []
    for k, sc in enumerate(scenarios):
        blk = pyo.Block()
        m.add_component(f"op_{k}", blk)
        p, q, tphi, pv = scenario_to_pu(net, sc)
        build_operation_block(blk, net, s, p, q, tphi, pv, cfg, scenario_hours(sc, cfg))
        m.ops.append(blk)
    m.first_stage_cost = pyo.Expression(expr=first_stage_cost(net, s, cfg))
    m.expected_operating_cost = pyo.Expression(
        expr=pyo.quicksum(sc.probability * m.ops[k].cost for k, sc in enumerate(scenarios)))
    m.obj = pyo.Objective(expr=m.first_stage_cost + m.expected_operating_cost, sense=pyo.minimize)
    return m


def make_solver(cfg: DNRConfig, time_limit: Optional[float] = None):
    """Create a configured solver object (solver name and options come from ``cfg``)."""
    opt = pyo.SolverFactory(cfg.solver)
    tl = time_limit if time_limit is not None else cfg.time_limit_s
    if cfg.solver.startswith("appsi"):
        opt.config.time_limit = tl
        opt.config.mip_gap = cfg.mip_gap
        opt.config.stream_solver = cfg.tee
        # appsi_highs does not reliably forward config.time_limit/mip_gap -> set HiGHS options directly
        opt.highs_options["time_limit"] = float(tl)
        opt.highs_options["mip_rel_gap"] = float(cfg.mip_gap)
        for k, v in cfg.solver_options.items():
            opt.highs_options[k] = v
    else:
        opt.options["time_limit" if "highs" in cfg.solver else "timelimit"] = tl
        opt.options["mip_rel_gap" if "highs" in cfg.solver else "mipgap"] = cfg.mip_gap
        for k, v in cfg.solver_options.items():
            opt.options[k] = v
    return opt


def solve_model(model: pyo.ConcreteModel, cfg: DNRConfig, time_limit: Optional[float] = None) -> Dict[str, Any]:
    """Solve ``model``; load the solution if feasible. Returns a status dictionary."""
    opt = make_solver(cfg, time_limit)
    t0 = time.perf_counter()
    kwargs = dict(load_solutions=False)
    if not cfg.solver.startswith("appsi"):
        kwargs["tee"] = cfg.tee
    res = opt.solve(model, **kwargs)
    elapsed = time.perf_counter() - t0
    term = str(res.solver.termination_condition)
    ok = term in ("optimal", "feasible", "maxTimeLimit") 
    loaded = False
    if ok:
        try:
            model.solutions.load_from(res)
            loaded = True
        except Exception:
            loaded = False
    ok = ok and loaded
    obj = pyo.value(model.obj) if loaded else None
    lb = getattr(res.problem, "lower_bound", None)
    gap = None
    if obj is not None and lb is not None and abs(obj) > 1e-9:
        gap = abs(obj - lb) / abs(obj)
    return dict(termination=term, status=str(res.solver.status), solve_time_s=elapsed,
                objective=obj, best_bound=lb, mip_gap=gap, solution_loaded=ok,
                optimal=(term == "optimal"))


def switch_values(model: pyo.ConcreteModel, net) -> Dict[int, int]:
    """Rounded first-stage switch statuses of a solved master model."""
    return {l: int(round(pyo.value(model.s[l]))) for l in net.branch_ids}


def evaluate_topology(net, switches: Mapping[int, int], scenarios: Sequence[Scenario],
                      cfg: DNRConfig) -> tuple[pyo.ConcreteModel, Dict[str, Any]]:
    """Solve the second-stage LP for a fixed topology. Returns (model, solve_info)."""
    m = build_stochastic_model(net, scenarios, cfg, fixed_switches=switches)
    info = solve_model(m, cfg)
    return m, info


def solve_stochastic_dnr(net, scenarios: Sequence[Scenario], cfg: DNRConfig):
    """Build and solve the stochastic DNR MILP. Returns (model, switches, solve_info)."""
    m = build_stochastic_model(net, scenarios, cfg)
    info = solve_model(m, cfg)
    sw = switch_values(m, net) if info["solution_loaded"] else None
    return m, sw, info


def diagnose_infeasibility(net, scenarios: Sequence[Scenario], cfg: DNRConfig) -> List[str]:
    """Explain why a model is infeasible (or what forces load shedding / violations).

    Steps: (1) first-stage feasibility (radiality + switching limits + fixed branches);
    (2) relax operation limits (shedding and curtailment allowed, hard voltage limits kept) and report which
    scenarios need how much relief.  Returns human-readable lines.
    """
    import dataclasses
    lines: List[str] = []
    m = pyo.ConcreteModel()
    build_first_stage(m, net, cfg)
    m.obj = pyo.Objective(expr=0.0)
    info = solve_model(m, cfg, time_limit=30)
    if info["termination"] == "infeasible":
        lines.append("First-stage constraints are infeasible: no radial spanning tree satisfies the "
                     "fixed (non-switchable) branches, max_switch_ops and exclusive_close_pairs. "
                     "Relax max_switch_ops / exclusive pairs or make more branches switchable.")
        return lines
    lines.append("First-stage constraints (radiality, switching limits) are feasible.")
    relaxed = dataclasses.replace(cfg, allow_shedding=True, allow_curtailment=True, soft_voltage=False)
    mm, sw, info2 = solve_stochastic_dnr(net, scenarios, relaxed)
    if not info2["solution_loaded"]:
        lines.append(f"Even the relaxed model failed ({info2['termination']}).")
        return lines
    for k, sc in enumerate(scenarios):
        blk = mm.ops[k]
        shed = net.pu_to_kw(pyo.value(blk.shed_pu))
        curt = net.pu_to_kw(pyo.value(blk.curt_pu))
        viol = pyo.value(blk.viol_pu)
        if shed > 1e-3 or curt > 1e-3 or viol > 1e-6:
            lines.append(f"Scenario {sc.scenario_id}: needs {shed:.1f} kW shedding, {curt:.1f} kW "
                         f"curtailment (voltage/ampacity limits bind in this scenario even with the best "
                         f"topology -> relax limits, add switchable ties/capacity, or allow shedding).")
    lines.append("Relaxed model switch plan: closed=" +
                 str([l for l in net.branch_ids if sw[l] == 1]))
    return lines
