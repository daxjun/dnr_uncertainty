"""Result extraction, validation report and CSV/JSON export.

Reported units: kW, kVAr, p.u. voltage magnitude, A, $.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import pyomo.environ as pyo

from models.distflow import DNRConfig
from utils.ac_power_flow import ac_power_flow
from utils.topology_validation import validate_topology


def _v(x) -> float:
    val = pyo.value(x, exception=False)
    return 0.0 if val is None else float(val)


def extract_solution(model, net, scenarios, switches: Mapping[int, int], cfg: DNRConfig,
                     info: Optional[dict] = None) -> Dict[str, Any]:
    """Collect the solution of a solved model (``model.ops`` blocks) into plain dicts."""
    closed = [l for l in net.branch_ids if switches[l] == 1]
    opened = [l for l in net.branch_ids if switches[l] == 0]
    ops_closed = [l for l in net.branch_ids if net.s0[l] == 0 and switches[l] == 1]
    ops_opened = [l for l in net.branch_ids if net.s0[l] == 1 and switches[l] == 0]
    out: Dict[str, Any] = dict(
        network=net.name, switches={str(l): int(switches[l]) for l in net.branch_ids},
        closed_branches=closed, open_branches=opened,
        switch_operations=dict(closed=ops_closed, opened=ops_opened,
                               n_operations=len(ops_closed) + len(ops_opened),
                               cost_usd=sum(net.switch_cost[l] for l in ops_closed + ops_opened)),
        topology=validate_topology(net, closed), scenarios={}, solver=info or {})
    ibase = net.i_base_a
    for k, sc in enumerate(scenarios):
        blk = model.ops[k]
        v = {i: math.sqrt(max(_v(blk.v2[i]), 0.0)) for i in net.bus_ids}
        P = {l: _v(blk.P[l]) for l in net.branch_ids}
        Q = {l: _v(blk.Q[l]) for l in net.branch_ids}
        I_a, loading = {}, {}
        for l in net.branch_ids:
            if switches[l] == 1:
                vi = max(v[net.from_bus[l]], 1e-6)
                I_a[l] = math.hypot(P[l], Q[l]) / vi * ibase
            else:
                I_a[l] = 0.0
            loading[l] = 100.0 * I_a[l] / net.i_max_a[l]
        loss_lin = net.pu_to_kw(_v(blk.loss_pu))
        loss_quad = net.pu_to_kw(sum(net.r_pu[l] * (P[l] ** 2 + Q[l] ** 2) for l in net.branch_ids))
        curt = {i: net.pu_to_kw(_v(blk.p_curt[i])) for i in blk.pv_buses}
        shed = {i: net.pu_to_kw(_v(blk.p_shed[i])) for i in blk.shed_buses}
        tot_load = sum(sc.p_load_kw.get(i, 0.0) for i in net.bus_ids if i != net.root_bus)
        root_p = sum(P[l] for l in net.lines_from[net.root_bus]) - sum(P[l] for l in net.lines_to[net.root_bus])
        out["scenarios"][sc.scenario_id] = dict(
            probability=sc.probability, period=sc.period,
            hours=sc.duration_h if sc.duration_h is not None else cfg.default_hours,
            cost_usd=_v(blk.cost), loss_kw_linearised=loss_lin, loss_kw_quadratic_post=loss_quad,
            curtailment_kw=curt, curtailment_total_kw=sum(curt.values()),
            pv_available_kw={i: sc.pv_kw.get(i, 0.0) for i in blk.pv_buses},
            load_shedding_kw=shed, load_shedding_total_kw=sum(shed.values()),
            load_total_kw=tot_load, substation_p_kw=net.pu_to_kw(root_p),
            voltage_pu=v, p_flow_kw={l: net.pu_to_kw(P[l]) for l in net.branch_ids},
            q_flow_kvar={l: net.pu_to_kw(Q[l]) for l in net.branch_ids},
            current_a=I_a, loading_pct=loading,
            soft_voltage_violation_pu2=_v(blk.viol_pu))
    probs = [sc.probability for sc in scenarios]
    sd = out["scenarios"]
    out["expected"] = dict(
        cost_usd=sum(p * sd[sc.scenario_id]["cost_usd"] for p, sc in zip(probs, scenarios)),
        loss_kw=sum(p * sd[sc.scenario_id]["loss_kw_linearised"] for p, sc in zip(probs, scenarios)),
        curtailment_kw=sum(p * sd[sc.scenario_id]["curtailment_total_kw"] for p, sc in zip(probs, scenarios)),
        load_shedding_kw=sum(p * sd[sc.scenario_id]["load_shedding_total_kw"] for p, sc in zip(probs, scenarios)))
    out["worst_case_cost_usd"] = max(d["cost_usd"] for d in sd.values())
    return out


def validate_solution(net, sol: Dict[str, Any], scenarios: Sequence, cfg: DNRConfig,
                      tol: float = 1e-5) -> Dict[str, Any]:
    """Automatic post-solve checks. Returns a report dict (also printable via print_report)."""
    checks: List[Dict[str, Any]] = []

    def add(name, passed, detail, severity="error"):
        checks.append(dict(name=name, passed=bool(passed), detail=detail, severity=severity))

    info = sol.get("solver", {})
    add("solver status optimal / acceptable", info.get("termination") in ("optimal", "feasible", "maxTimeLimit")
        and info.get("solution_loaded", True), f"termination={info.get('termination')}, gap={info.get('mip_gap')}")
    topo = sol["topology"]
    add("topology connected (all buses reachable from substation)", topo["is_connected"],
        f"{topo['n_energized_buses']}/{len(net.bus_ids)} buses reachable")
    add("no cycles in energised network", not topo["has_cycle"], f"independent loops={topo['n_independent_loops']}")
    add("closed branches = N-1", topo["n_closed_branches"] == topo["expected_branches"],
        f"{topo['n_closed_branches']} closed vs {topo['expected_branches']}")
    add("radial tree structure", topo["is_radial"], topo["message"])
    closed = set(sol["closed_branches"])
    worst = dict(bal=0.0, vlo=0.0, vhi=0.0, load=0.0, flow_open=0.0, ac_dv=0.0, ac_loss=0.0, ac_load=0.0,
                 ac_vviol=0.0, neg=0.0, bounds=0.0)
    for sc in scenarios:
        d = sol["scenarios"][sc.scenario_id]
        # power balance residual (kW) recomputed from reported flows
        for b in net.bus_ids:
            if b == net.root_bus:
                continue
            net_in = sum(d["p_flow_kw"][l] for l in net.lines_to[b]) - sum(d["p_flow_kw"][l] for l in net.lines_from[b])
            dem = sc.p_load_kw.get(b, 0.0) - d["load_shedding_kw"].get(b, 0.0) \
                - (sc.pv_kw.get(b, 0.0) - d["curtailment_kw"].get(b, 0.0))
            worst["bal"] = max(worst["bal"], abs(net_in - dem))
        for b, v in d["voltage_pu"].items():
            worst["vlo"] = max(worst["vlo"], net.vmin[b] - v)
            worst["vhi"] = max(worst["vhi"], v - net.vmax[b])
        worst["load"] = max(worst["load"], max(d["loading_pct"].values()) - 100.0)
        worst["flow_open"] = max([worst["flow_open"]] + [abs(d["p_flow_kw"][l]) + abs(d["q_flow_kvar"][l])
                                                          for l in net.branch_ids if l not in closed])
        for b, c in d["curtailment_kw"].items():
            worst["neg"] = max(worst["neg"], -c)
            worst["bounds"] = max(worst["bounds"], c - sc.pv_kw.get(b, 0.0))
        for b, c in d["load_shedding_kw"].items():
            worst["neg"] = max(worst["neg"], -c)
            worst["bounds"] = max(worst["bounds"], c - sc.p_load_kw.get(b, 0.0))
        # nonlinear AC check
        pn, qn = {}, {}
        for b in net.bus_ids:
            if b == net.root_bus:
                continue
            sh = d["load_shedding_kw"].get(b, 0.0)
            pl = sc.p_load_kw.get(b, 0.0)
            ql = sc.q_load_kvar.get(b, 0.0)
            used = sc.pv_kw.get(b, 0.0) - d["curtailment_kw"].get(b, 0.0)
            pn[b] = net.kw_to_pu(pl - sh - used)
            qn[b] = net.kw_to_pu(ql - (ql / pl * sh if pl > 1e-9 else 0.0))
        ac = ac_power_flow(net, sol["closed_branches"], pn, qn)
        d["ac_validation"] = dict(converged=ac["converged"], v_min_pu=min(ac["v"].values()),
                                  v_max_pu=max(ac["v"].values()),
                                  max_abs_voltage_error_pu=max(abs(ac["v"][b] - d["voltage_pu"][b]) for b in ac["v"]),
                                  loss_kw_ac=net.pu_to_kw(ac["loss_pu"]),
                                  loss_error_kw=net.pu_to_kw(ac["loss_pu"]) - d["loss_kw_linearised"],
                                  max_current_loading_pct=max(100 * ac["i_pu"][l] * net.i_base_a / net.i_max_a[l]
                                                              for l in ac["i_pu"]))
        a = d["ac_validation"]
        worst["ac_dv"] = max(worst["ac_dv"], a["max_abs_voltage_error_pu"])
        worst["ac_loss"] = max(worst["ac_loss"], abs(a["loss_error_kw"]))
        worst["ac_load"] = max(worst["ac_load"], a["max_current_loading_pct"] - 100.0)
        worst["ac_vviol"] = max(worst["ac_vviol"], max(
            max(net.vmin[b] - v for b, v in ac["v"].items()), max(v - net.vmax[b] for b, v in ac["v"].items())))
    kw_tol = tol * net.s_base_mva * 1000.0
    add("power-balance residuals within tolerance", worst["bal"] <= max(kw_tol, 1e-3),
        f"max residual {worst['bal']:.2e} kW")
    soft = cfg.soft_voltage
    add("voltage limits respected (MILP)", soft or (worst["vlo"] <= 1e-6 and worst["vhi"] <= 1e-6),
        f"max violation low={max(worst['vlo'], 0):.2e}, high={max(worst['vhi'], 0):.2e} p.u.")
    add("line limits respected (MILP current at actual voltage)", worst["load"] <= 1.0,
        f"max overload {max(worst['load'], 0):.3f} % of limit")
    add("no flow on open branches (same switch plan in every scenario)", worst["flow_open"] <= 1e-3,
        f"max |flow| on open branches {worst['flow_open']:.2e} kW/kVAr; switch plan is one first-stage vector")
    add("PV curtailment / load shedding nonnegative and within bounds",
        worst["neg"] <= 1e-6 and worst["bounds"] <= 1e-4, f"min {-worst['neg']:.2e}, max excess {worst['bounds']:.2e} kW")
    add("AC power-flow converged", all(d["ac_validation"]["converged"] for d in sol["scenarios"].values()), "backward/forward sweep")
    add("AC vs LinDistFlow voltage error small (< 0.01 p.u.)", worst["ac_dv"] < 0.01,
        f"max |dV| = {worst['ac_dv']:.4f} p.u.", "warning")
    add("AC voltage limits respected", worst["ac_vviol"] <= 1e-4, f"max AC violation {max(worst['ac_vviol'], 0):.4f} p.u.", "warning")
    add("AC branch currents within limit", worst["ac_load"] <= 1.0, f"max AC overload {max(worst['ac_load'], 0):.2f} %", "warning")
    add("max loss error |AC - linearised| reported", True, f"{worst['ac_loss']:.2f} kW (linearised loss is a lower approximation)", "info")
    shed_tot = sum(d["load_shedding_total_kw"] for d in sol["scenarios"].values())
    add("load shedding only if necessary (zero in all scenarios?)", True,
        f"total shed over scenarios = {shed_tot:.3f} kW", "info")
    all_ok = all(c["passed"] for c in checks if c["severity"] == "error")
    return dict(all_passed=all_ok, n_checks=len(checks), checks=checks)


def print_report(report: Dict[str, Any], title: str = "VALIDATION REPORT") -> None:
    """Print a validation report to the console."""
    print(f"\n=== {title} ===")
    for c in report["checks"]:
        tag = "PASS" if c["passed"] else ("FAIL" if c["severity"] == "error" else "WARN")
        if c["severity"] == "info":
            tag = "INFO"
        print(f"[{tag}] {c['name']}: {c['detail']}")
    print("OVERALL:", "ALL CRITICAL CHECKS PASSED" if report["all_passed"] else "SOME CRITICAL CHECKS FAILED")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def save_results(sol: Dict[str, Any], report: Dict[str, Any], out_dir: Path, prefix: str,
                 json_name: str) -> None:
    """Write JSON (solution + validation) and CSV tables (switches, scenario summary, voltages, flows)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(solution=sol, validation=report)
    (out_dir / json_name).write_text(json.dumps(_jsonable(payload), indent=2))
    (out_dir / f"{prefix}_validation_report.json").write_text(json.dumps(_jsonable(report), indent=2))
    pd.DataFrame(dict(branch_id=[int(k) for k in sol["switches"]],
                      closed=list(sol["switches"].values()))).to_csv(out_dir / f"{prefix}_switches.csv", index=False)
    rows, vrows, frows = [], [], []
    for sid, d in sol["scenarios"].items():
        rows.append(dict(scenario=sid, probability=d["probability"], cost_usd=d["cost_usd"],
                         loss_kw=d["loss_kw_linearised"], loss_kw_quadratic=d["loss_kw_quadratic_post"],
                         loss_kw_ac=d["ac_validation"]["loss_kw_ac"] if "ac_validation" in d else None,
                         curtailment_kw=d["curtailment_total_kw"], load_shedding_kw=d["load_shedding_total_kw"],
                         load_kw=d["load_total_kw"], substation_p_kw=d["substation_p_kw"],
                         v_min_pu=min(d["voltage_pu"].values()), v_max_pu=max(d["voltage_pu"].values())))
        for b, v in d["voltage_pu"].items():
            vrows.append(dict(scenario=sid, bus=b, v_pu=v))
        for l in d["p_flow_kw"]:
            frows.append(dict(scenario=sid, branch=l, p_kw=d["p_flow_kw"][l], q_kvar=d["q_flow_kvar"][l],
                              current_a=d["current_a"][l], loading_pct=d["loading_pct"][l]))
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_scenario_summary.csv", index=False)
    pd.DataFrame(vrows).to_csv(out_dir / f"{prefix}_bus_voltages.csv", index=False)
    pd.DataFrame(frows).to_csv(out_dir / f"{prefix}_branch_flows.csv", index=False)
