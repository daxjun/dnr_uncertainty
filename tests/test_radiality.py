"""Radiality: validation function and MILP spanning-tree formulations."""
import networkx as nx
import pyomo.environ as pyo
import pytest

from models.distflow import DNRConfig
from models.radiality import add_radiality_constraints, add_switch_variables
from models.stochastic_dnr import make_solver
from tests.helpers import small_network
from utils.topology_validation import validate_topology

pytest.importorskip("highspy")


def test_validate_accepts_tree():
    net = small_network()
    r = validate_topology(net, [1, 2, 3, 4, 5])
    assert r["is_radial"] and r["is_connected"] and not r["has_cycle"]


def test_validate_rejects_loop_plus_island_with_correct_count():
    """Branch count alone (N-1 = 5) is satisfied but topology is invalid."""
    net = small_network()
    r = validate_topology(net, [2, 3, 7, 6, 5])      # loop 3-4-6-5-3 plus isolated root bus 1
    assert r["n_closed_branches"] == 5
    assert not r["is_radial"]
    assert r["unreachable_buses"] or r["has_cycle"]


def test_validate_detects_disconnection_and_cycle_separately():
    net = small_network()
    assert not validate_topology(net, [1, 2, 3, 4])["is_connected"]
    r = validate_topology(net, [1, 2, 3, 4, 5, 6])
    assert r["has_cycle"] and not r["is_radial"]


def _tree_model(method, weights, fix=None):
    net = small_network()
    m = pyo.ConcreteModel()
    add_switch_variables(m, net)
    add_radiality_constraints(m, net, method)
    if fix:
        for l, v in fix.items():
            m.s[l].fix(v)
    m.obj = pyo.Objective(expr=sum(weights[l] * m.s[l] for l in net.branch_ids))
    return net, m


@pytest.mark.parametrize("method", ["scf", "directed"])
def test_milp_equals_minimum_spanning_tree(method):
    w = {1: 5.0, 2: 3.0, 3: 7.0, 4: 2.0, 5: 4.5, 6: 1.1, 7: 6.2}
    net, m = _tree_model(method, w)
    res = make_solver(DNRConfig()).solve(m, load_solutions=True)
    closed = [l for l in net.branch_ids if pyo.value(m.s[l]) > 0.5]
    assert validate_topology(net, closed)["is_radial"]
    g = nx.Graph()
    for l in net.branch_ids:
        g.add_edge(net.from_bus[l], net.to_bus[l], weight=w[l])
    mst = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(g, data=True))
    assert pyo.value(m.obj) == pytest.approx(mst, rel=1e-6)


@pytest.mark.parametrize("method", ["scf", "directed"])
def test_loop_plus_island_is_infeasible_in_milp(method):
    """Pattern with N-1 closed branches but a loop and an island must be cut off."""
    fix = {l: 0 for l in range(1, 8)}
    fix.update({2: 1, 3: 1, 7: 1, 6: 1, 5: 1})   # 5 closed branches, loop {3,4,6,5}, root island
    net, m = _tree_model(method, {l: 1.0 for l in range(1, 8)}, fix=fix)
    res = make_solver(DNRConfig()).solve(m, load_solutions=False)
    assert str(res.solver.termination_condition) in ("infeasible", "infeasibleOrUnbounded")
