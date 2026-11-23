"""Two-stage robust distribution network reconfiguration with budgeted PV/load uncertainty.

Usage:  python main_robust.py [--method ccg|box] [--oracle heuristic|milp] [--gamma-load 6] ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyomo.environ as pyo

from data.network_data import load_example_network, load_network_csv, load_network_excel
from models.distflow import DNRConfig
from models.robust_dnr import (UncertaintySet, sample_uncertainty_set, solve_robust_box, solve_robust_ccg)
from models.stochastic_dnr import evaluate_topology
from utils.plotting import make_all_plots, plot_ccg_convergence
from utils.results_processing import extract_solution, print_report, save_results, validate_solution
from utils.scenario_generation import forecast_points

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network-excel"); ap.add_argument("--branches-csv"); ap.add_argument("--buses-csv")
    ap.add_argument("--meta-json")
    ap.add_argument("--periods", type=int, nargs="+", default=[12, 19], help="forecast hours (0-23)")
    ap.add_argument("--method", default="ccg", choices=["ccg", "box"])
    ap.add_argument("--oracle", default="heuristic", choices=["heuristic", "milp"])
    ap.add_argument("--load-dev", type=float, default=0.15)
    ap.add_argument("--pv-dev", type=float, default=0.30)
    ap.add_argument("--gamma-load", type=int, default=6)
    ap.add_argument("--gamma-pv", type=int, default=2)
    ap.add_argument("--pv-two-sided", action="store_true")
    ap.add_argument("--radiality", default="scf", choices=["scf", "directed"])
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--max-iter", type=int, default=15)
    ap.add_argument("--max-switch-ops", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    if a.network_excel:
        net = load_network_excel(a.network_excel)
    elif a.branches_csv and a.buses_csv:
        net = load_network_csv(a.branches_csv, a.buses_csv, a.meta_json)
    else:
        net = load_example_network()
    cfg = DNRConfig(solver=a.solver, time_limit_s=a.time_limit, radiality_method=a.radiality,
                    max_switch_ops=a.max_switch_ops)
    uset = UncertaintySet(load_dev=a.load_dev, pv_dev=a.pv_dev, gamma_load=a.gamma_load,
                          gamma_pv=a.gamma_pv, pv_two_sided=a.pv_two_sided)
    forecasts = forecast_points(net, a.periods)
    print(f"Network: {net.name}; periods {a.periods}; load +-{a.load_dev:.0%} (Gamma_L={a.gamma_load}), "
          f"PV -{a.pv_dev:.0%}{'/+' if a.pv_two_sided else ''} (Gamma_PV={a.gamma_pv})")

    if a.method == "ccg":
        res = solve_robust_ccg(net, forecasts, uset, cfg, max_iter=a.max_iter, oracle=a.oracle)
    else:
        res = solve_robust_box(net, forecasts, uset, cfg)
    print(f"Method: {res['method']} | converged={res['converged']} | exact={res['exact']} | {res['time_s']:.1f} s")
    sw = res["switches"]

    # Detailed evaluation of the final topology: forecasts, worst-case realisations, random vertices of U.
    from data.load_scenarios import Scenario
    samples, sample_period = [], []
    for f in forecasts:
        smp = sample_uncertainty_set(net, f, uset, n=8)
        samples += smp
        sample_period += [f.period] * len(smp)
    # report scenarios: forecasts + worst-case realisations (probability = period weight / 2 each)
    rep_sc = []
    for f in forecasts:
        rep_sc.append(Scenario(f.scenario_id, f.probability / 2, f.p_load_kw, f.q_load_kvar, f.pv_kw, f.period))
    for w in res["worst_realizations"]:
        rep_sc.append(Scenario(w.scenario_id, w.probability / 2, w.p_load_kw, w.q_load_kvar, w.pv_kw, w.period,
                               meta=w.meta))
    model, info = evaluate_topology(net, sw, rep_sc, cfg)
    info["objective_robust_master"] = res["upper_bound"]
    sol = extract_solution(model, net, rep_sc, sw, cfg, info)

    # sampling check: no random vertex of U may cost more than the worst realisation found
    smp_model, smp_info = evaluate_topology(
        net, sw, [Scenario(f"s{k}", 1.0 / len(samples), s.p_load_kw, s.q_load_kvar, s.pv_kw, s.period)
                  for k, s in enumerate(samples)], cfg)
    smp_costs = [float(pyo.value(b.cost)) for b in smp_model.ops]
    smp_shed = [net.pu_to_kw(float(pyo.value(b.shed_pu))) for b in smp_model.ops]
    worst_found = {}
    for f, q in zip(forecasts, res.get("worst_costs", [None] * len(forecasts))):
        worst_found[f.period] = q
    sol["robust"] = dict(
        method=res["method"], exact=res["exact"], converged=res["converged"],
        uncertainty_set=res["uncertainty_set"], lower_bound_usd=res["lower_bound"],
        worst_case_total_cost_usd=res["worst_case_total_cost"], solve_time_s=res["time_s"],
        worst_cost_by_period_usd=worst_found, history=res["history"],
        sampling_check=dict(n_samples=len(samples), max_sample_cost_usd=max(smp_costs),
                            max_sample_shed_kw=max(smp_shed)),
        note=("Heuristic separation: protection is guaranteed against every generated realisation; "
              "a worse realisation inside U may exist." if res["method"].startswith("C&CG with heuristic") else
              ("Conservative approximation; not exact for the budgeted set." if not res["exact"] else
               "Exact: MILP oracle converged.")))
    exceed = 0.0
    for c, per in zip(smp_costs, sample_period):
        wf = worst_found.get(per)
        if wf:
            exceed = max(exceed, (c - wf) / max(wf, 1e-9))
    report = validate_solution(net, sol, rep_sc, cfg)
    # extra robust checks
    robust_checks = [
        dict(name="no load shedding at any tested realisation inside U",
             passed=max(smp_shed) <= 1e-3 and sol["expected"]["load_shedding_kw"] <= 1e-3,
             detail=f"max shed over {len(samples)} random vertices = {max(smp_shed):.3f} kW", severity="warning"),
        dict(name="random vertices of U do not exceed the worst-case cost found (+1 %)",
             passed=exceed <= 1e-2, detail=f"max relative excess of sampled period cost over worst found = "
             f"{exceed:.2%} (worst found per period: {[round(v, 2) for v in worst_found.values() if v is not None]})",
             severity="warning")]
    report["checks"].extend(robust_checks)
    print("\nOpened switches (initially closed):", sol["switch_operations"]["opened"])
    print("Closed switches (initially open)  :", sol["switch_operations"]["closed"])
    print("Open branches in final topology    :", sol["open_branches"])
    print(f"Robust objective (switching + weighted worst-case cost): {res['upper_bound']:.2f} $ "
          f"(master lower bound {res['lower_bound']})")
    print(f"{'scenario':20s}{'cost $':>10s}{'loss kW':>10s}{'curt kW':>10s}{'shed kW':>10s}{'Vmin':>8s}{'Vmax':>8s}")
    for sid, d in sol["scenarios"].items():
        print(f"{sid:20s}{d['cost_usd']:10.2f}{d['loss_kw_linearised']:10.2f}{d['curtailment_total_kw']:10.2f}"
              f"{d['load_shedding_total_kw']:10.2f}{min(d['voltage_pu'].values()):8.4f}{max(d['voltage_pu'].values()):8.4f}")
    print_report(report)
    out = Path(a.out_dir)
    save_results(sol, report, out, "robust", "robust_solution.json")
    if not a.no_plots:
        for p in make_all_plots(net, sol, out / "figures", "robust"):
            print("figure:", p)
        pc = plot_ccg_convergence(res["history"], out / "figures" / "robust_ccg_convergence.png")
        if pc:
            print("figure:", pc)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
