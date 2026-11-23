"""Stochastic model, scenario generation and (small-feeder) robust C&CG."""
import pyomo.environ as pyo
import pytest

from data.load_scenarios import (normalize_probabilities, read_scenarios, scenarios_to_dataframe,
                                 validate_scenarios, write_scenarios_csv)
from data.network_data import load_example_network
from models.distflow import DNRConfig
from models.robust_dnr import UncertaintySet, solve_robust_box, solve_robust_ccg
from models.stochastic_dnr import build_stochastic_model, evaluate_topology, solve_stochastic_dnr
from tests.helpers import scen_from_net, small_network
from utils.results_processing import extract_solution, validate_solution
from utils.scenario_generation import (forecast_points, generate_random_scenarios,
                                       generate_structured_scenarios)

pytest.importorskip("highspy")


# ---------------------------------------------------------------- scenario generation
def test_structured_scenarios_probabilities_and_bounds():
    net = load_example_network()
    sc = generate_structured_scenarios(net)
    assert len(sc) == 6 and sum(s.probability for s in sc) == pytest.approx(1.0)
    validate_scenarios(net, sc)
    adv = next(s for s in sc if s.scenario_id == "h12_adverse")
    nom = next(s for s in sc if s.scenario_id == "h12_nominal")
    assert adv.p_load_kw[24] == pytest.approx(1.15 * nom.p_load_kw[24])
    assert adv.pv_kw[25] == pytest.approx(0.70 * nom.pv_kw[25])


def test_random_scenarios_reproducible_and_respect_capacity():
    net = load_example_network()
    a = generate_random_scenarios(net, 4, seed=5)
    b = generate_random_scenarios(net, 4, seed=5)
    c = generate_random_scenarios(net, 4, seed=6)
    assert [s.p_load_kw[10] for s in a] == [s.p_load_kw[10] for s in b]
    assert [s.p_load_kw[10] for s in a] != [s.p_load_kw[10] for s in c]
    validate_scenarios(net, a)


def test_user_probabilities_and_csv_roundtrip(tmp_path):
    net = load_example_network()
    sc = generate_random_scenarios(net, 2, seed=1, probabilities=[1, 2, 3, 4])
    assert [round(s.probability, 3) for s in sc] == [0.1, 0.2, 0.3, 0.4]
    p = tmp_path / "s.csv"
    write_scenarios_csv(sc, p)
    back = normalize_probabilities(read_scenarios(p))
    assert [s.scenario_id for s in back] == [s.scenario_id for s in sc]
    assert back[2].p_load_kw[7] == pytest.approx(sc[2].p_load_kw[7])
    assert back[3].probability == pytest.approx(0.4)


# ------------------------------------------------------------------ stochastic model
def test_stochastic_model_single_switch_plan_and_expected_cost():
    net = small_network(pv={4: 800.0})
    scen = [scen_from_net(net, "a", 0.3, 1.2, 0.5), scen_from_net(net, "b", 0.7, 0.8, 1.0)]
    cfg = DNRConfig(time_limit_s=60)
    m, sw, info = solve_stochastic_dnr(net, scen, cfg)
    assert info["optimal"]
    sol = extract_solution(m, net, scen, sw, cfg, info)
    rep = validate_solution(net, sol, scen, cfg)
    assert rep["all_passed"], [c for c in rep["checks"] if not c["passed"]]
    assert sol["topology"]["is_radial"]
    total = pyo.value(m.first_stage_cost) + sum(s.probability * sol["scenarios"][s.scenario_id]["cost_usd"] for s in scen)
    assert total == pytest.approx(info["objective"], rel=1e-7)
    # same switch vector is used in all blocks: no flow outside the closed set in any scenario
    for d in sol["scenarios"].values():
        for l in sol["open_branches"]:
            assert abs(d["p_flow_kw"][l]) < 1e-6


def test_both_radiality_formulations_agree():
    net = small_network()
    scen = [scen_from_net(net, "a", 0.5, 1.1), scen_from_net(net, "b", 0.5, 0.9)]
    objs = []
    for method in ("scf", "directed"):
        _, _, info = solve_stochastic_dnr(net, scen, DNRConfig(radiality_method=method))
        objs.append(info["objective"])
    assert objs[0] == pytest.approx(objs[1], rel=1e-6)


def test_max_switch_ops_and_exclusive_pairs_respected():
    net = small_network()
    scen = [scen_from_net(net, "a", 1.0, 1.0)]
    m, sw, info = solve_stochastic_dnr(net, scen, DNRConfig(max_switch_ops=0))
    assert all(sw[l] == net.s0[l] for l in net.branch_ids)
    m, sw, info = solve_stochastic_dnr(net, scen, DNRConfig(exclusive_close_pairs=[(6, 7)]))
    assert not (sw[6] == 1 and sw[7] == 1)


# ----------------------------------------------------------------------------- robust
def _robust_setup():
    net = small_network(pv={4: 800.0})
    fc = [scen_from_net(net, "f", 1.0, 1.0, 0.6)]
    fc[0].period = 12
    return net, fc


def test_robust_ccg_milp_oracle_exact_on_small_feeder():
    net, fc = _robust_setup()
    cfg = DNRConfig(time_limit_s=60)
    u = UncertaintySet(load_dev=0.2, pv_dev=0.3, gamma_load=2, gamma_pv=1)
    res = solve_robust_ccg(net, fc, u, cfg, max_iter=8, oracle="milp", oracle_time=60, verbose=False)
    assert res["converged"] and res["exact"]
    assert res["lower_bound"] <= res["upper_bound"] + 1e-6
    assert res["upper_bound"] - res["lower_bound"] <= 2e-3 * max(1.0, res["upper_bound"])
    # exact worst case >= nominal cost
    m0, _ = evaluate_topology(net, res["switches"], fc, cfg)
    assert res["worst_costs"][0] >= pyo.value(m0.ops[0].cost) - 1e-9


def test_heuristic_and_milp_oracles_agree_on_small_feeder():
    net, fc = _robust_setup()
    cfg = DNRConfig(time_limit_s=60)
    u = UncertaintySet(load_dev=0.2, pv_dev=0.3, gamma_load=2, gamma_pv=1)
    r1 = solve_robust_ccg(net, fc, u, cfg, oracle="milp", oracle_time=60, verbose=False)
    r2 = solve_robust_ccg(net, fc, u, cfg, oracle="heuristic", verbose=False)
    assert r2["upper_bound"] == pytest.approx(r1["upper_bound"], rel=5e-3)


def test_box_approximation_flagged_not_exact():
    net, fc = _robust_setup()
    res = solve_robust_box(net, fc, UncertaintySet(gamma_load=2, gamma_pv=1), DNRConfig(), verbose=False)
    assert res["exact"] is False and "NOT exact" in res["method"]
