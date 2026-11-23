"""Two-stage stochastic distribution network reconfiguration (IEEE 33-bus demo).

Usage:  python main_stochastic.py [--scenarios-csv FILE] [--radiality scf|directed] ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data.load_scenarios import read_scenarios, normalize_probabilities, write_scenarios_csv
from data.network_data import load_example_network, load_network_csv, load_network_excel
from models.distflow import DNRConfig
from models.stochastic_dnr import diagnose_infeasibility, solve_stochastic_dnr
from utils.plotting import make_all_plots
from utils.results_processing import extract_solution, print_report, save_results, validate_solution
from utils.scenario_generation import generate_random_scenarios, generate_structured_scenarios

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network-excel", help="Excel workbook (sheets buses, branches, meta)")
    ap.add_argument("--branches-csv"); ap.add_argument("--buses-csv"); ap.add_argument("--meta-json")
    ap.add_argument("--scenarios-csv", help="CSV/Excel scenario table (see README); default: generated")
    ap.add_argument("--random-scenarios", type=int, default=0, help="N random scenarios per period instead of the 3-case tree")
    ap.add_argument("--radiality", default="scf", choices=["scf", "directed"])
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--max-switch-ops", type=int, default=None)
    ap.add_argument("--soft-voltage", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    if a.network_excel:
        net = load_network_excel(a.network_excel)
    elif a.branches_csv and a.buses_csv:
        net = load_network_csv(a.branches_csv, a.buses_csv, a.meta_json)
    else:
        net = load_example_network()
    print(f"Network: {net.name} | {len(net.bus_ids)} buses, {len(net.branch_ids)} branches | "
          f"S_base={net.s_base_mva} MVA, V_base={net.v_base_kv} kV")

    if a.scenarios_csv:
        scenarios = normalize_probabilities(read_scenarios(a.scenarios_csv))
    elif a.random_scenarios > 0:
        scenarios = generate_random_scenarios(net, n_per_period=a.random_scenarios)
    else:
        scenarios = generate_structured_scenarios(net)
    print(f"{len(scenarios)} scenarios, probabilities: " +
          ", ".join(f"{s.scenario_id}={s.probability:.3f}" for s in scenarios))

    cfg = DNRConfig(solver=a.solver, time_limit_s=a.time_limit, radiality_method=a.radiality,
                    max_switch_ops=a.max_switch_ops, soft_voltage=a.soft_voltage)
    model, switches, info = solve_stochastic_dnr(net, scenarios, cfg)
    print(f"Solver: {info['termination']} in {info['solve_time_s']:.2f} s, objective = {info['objective']}")
    if switches is None:
        print("NO SOLUTION. Diagnostics:")
        for line in diagnose_infeasibility(net, scenarios, cfg):
            print("  -", line)
        raise SystemExit(1)

    sol = extract_solution(model, net, scenarios, switches, cfg, info)
    sol["model"] = dict(type="two-stage stochastic MILP (LinDistFlow)", radiality=a.radiality,
                        n_scenarios=len(scenarios), first_stage_cost_usd=sol["switch_operations"]["cost_usd"])
    report = validate_solution(net, sol, scenarios, cfg)
    print("\nOpened switches (initially closed):", sol["switch_operations"]["opened"])
    print("Closed switches (initially open)  :", sol["switch_operations"]["closed"])
    print("Open branches in final topology    :", sol["open_branches"])
    print(f"Expected operating cost: {sol['expected']['cost_usd']:.2f} $ | expected loss "
          f"{sol['expected']['loss_kw']:.2f} kW | curtailment {sol['expected']['curtailment_kw']:.2f} kW | "
          f"shedding {sol['expected']['load_shedding_kw']:.2f} kW")
    print(f"{'scenario':16s}{'prob':>7s}{'loss kW':>10s}{'curt kW':>10s}{'shed kW':>10s}{'Vmin':>8s}{'Vmax':>8s}")
    for sid, d in sol["scenarios"].items():
        print(f"{sid:16s}{d['probability']:7.3f}{d['loss_kw_linearised']:10.2f}{d['curtailment_total_kw']:10.2f}"
              f"{d['load_shedding_total_kw']:10.2f}{min(d['voltage_pu'].values()):8.4f}{max(d['voltage_pu'].values()):8.4f}")
    print_report(report)

    out = Path(a.out_dir)
    save_results(sol, report, out, "stochastic", "stochastic_solution.json")
    write_scenarios_csv(scenarios, out / "stochastic_scenarios_used.csv")
    if not a.no_plots:
        for p in make_all_plots(net, sol, out / "figures", "stochastic"):
            print("figure:", p)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
