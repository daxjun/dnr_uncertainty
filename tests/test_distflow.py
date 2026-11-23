"""LinDistFlow block: balance, voltage-drop equation, PV reverse flow, AC comparison."""
import math

import pyomo.environ as pyo
import pytest

from models.distflow import DNRConfig
from models.stochastic_dnr import build_stochastic_model, solve_model
from tests.helpers import scen_from_net, small_network
from utils.results_processing import extract_solution, validate_solution

pytest.importorskip("highspy")
FIXED = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0}


def _solve(net, sc, cfg=None):
    cfg = cfg or DNRConfig()
    m = build_stochastic_model(net, [sc], cfg, fixed_switches=FIXED)
    info = solve_model(m, cfg)
    assert info["optimal"]
    return m, cfg, info


def test_flows_equal_downstream_load_and_voltage_drop():
    net = small_network()
    sc = scen_from_net(net)
    m, cfg, _ = _solve(net, sc)
    b = m.ops[0]
    # branch 3 (3->4) carries bus-4 load; branch 2 (2->3) carries buses 3+4
    assert pyo.value(b.P[3]) == pytest.approx(net.kw_to_pu(500), abs=1e-9)
    assert pyo.value(b.P[2]) == pytest.approx(net.kw_to_pu(900), abs=1e-9)
    assert pyo.value(b.P[1]) == pytest.approx(net.kw_to_pu(1200), abs=1e-9)
    for l in (1, 2, 3, 4, 5):
        i, j = net.from_bus[l], net.to_bus[l]
        drop = pyo.value(b.v2[i]) - pyo.value(b.v2[j])
        exp = 2 * (net.r_pu[l] * pyo.value(b.P[l]) + net.x_pu[l] * pyo.value(b.Q[l]))
        assert drop == pytest.approx(exp, abs=1e-9)


def test_loss_envelope_is_lower_bound_of_quadratic_loss():
    net = small_network()
    m, cfg, _ = _solve(net, scen_from_net(net))
    b = m.ops[0]
    lin = pyo.value(b.loss_pu)
    quad = sum(net.r_pu[l] * (pyo.value(b.P[l]) ** 2 + pyo.value(b.Q[l]) ** 2) for l in net.branch_ids)
    assert lin <= quad + 1e-12 and lin > 0.9 * quad


def test_pv_reverse_flow_and_balance_residual_and_ac_agreement():
    net = small_network(pv={4: 1500.0})
    sc = scen_from_net(net, pv_f=0.8)
    m, cfg, info = _solve(net, sc)
    sol = extract_solution(m, net, [sc], FIXED, cfg, info)
    d = sol["scenarios"]["s"]
    assert d["p_flow_kw"][3] < 0      # PV (1200) exceeds bus-4 load (500)
    rep = validate_solution(net, sol, [sc], cfg)
    assert rep["all_passed"], [c for c in rep["checks"] if not c["passed"]]
    assert d["ac_validation"]["max_abs_voltage_error_pu"] < 0.01


def test_overvoltage_forces_curtailment_only_when_needed():
    net = small_network(pv={4: 3000.0}, v_root=1.04)
    sc = scen_from_net(net, pv_f=1.0)
    m, cfg, info = _solve(net, sc)
    sol = extract_solution(m, net, [sc], FIXED, cfg, info)
    assert sol["scenarios"]["s"]["curtailment_total_kw"] > 1.0
    assert max(sol["scenarios"]["s"]["voltage_pu"].values()) <= 1.05 + 1e-9
    net2 = small_network(pv={4: 300.0})
    m2, cfg2, info2 = _solve(net2, scen_from_net(net2))
    assert extract_solution(m2, net2, [scen_from_net(net2)], FIXED, cfg2, info2)["scenarios"]["s"]["curtailment_total_kw"] < 1e-6


def test_shedding_used_only_when_voltage_infeasible():
    net = small_network(loads={2: 300, 3: 400, 4: 500, 5: 300, 6: 400})
    m, cfg, info = _solve(net, scen_from_net(net))
    assert extract_solution(m, net, [scen_from_net(net)], FIXED, cfg, info)["scenarios"]["s"]["load_shedding_total_kw"] < 1e-6
    heavy = scen_from_net(net, load_f=12.0)   # impossible without shedding
    m2, cfg2, info2 = _solve(net, heavy)
    assert extract_solution(m2, net, [heavy], FIXED, cfg2, info2)["scenarios"]["s"]["load_shedding_total_kw"] > 1.0
