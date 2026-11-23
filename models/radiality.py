"""First-stage (here-and-now) switching decisions and radiality constraints.

Switch variable  s[l] in {0,1}  (1 = closed) -- a FIRST-STAGE variable, shared by all scenarios.

Radiality (spanning tree over *all* buses, rooted at the substation)
--------------------------------------------------------------------
Two rigorous formulations are implemented; both guarantee connectivity AND absence of loops:

``"scf"`` (default) -- single-commodity flow + edge count
    The root injects N-1 units of a fictitious commodity; every non-root bus consumes 1 unit.
    Commodity may only travel on closed branches (f_l^+ + f_l^- <= (N-1) s_l), so every bus is
    connected to the root through closed branches (no islands).  Together with
    sum_l s_l = N-1 a connected graph on N buses is a tree (no loops).
    NOTE: the count equality alone would NOT be sufficient (a loop plus an island satisfies it);
    it is the flow that rules the island out.

``"directed"`` -- Jabr-style directed spanning tree (parent arcs) + commodity flow
    Binary parent arcs y_l^+ (from->to) and y_l^- (to->from), s_l = y_l^+ + y_l^-.
    Every non-root bus has exactly one parent arc, the root has none.  Parent-arc cycles that
    are detached from the root are eliminated by the commodity flow (f_arc <= (N-1) y_arc).
    This formulation has a tighter LP relaxation (Jabr, Singh & Pal, IEEE TPWRS 2012).

The commodity-flow variables are first-stage variables: topology does not depend on scenarios.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

import pyomo.environ as pyo

RADIALITY_METHODS = ("scf", "directed")


def add_switch_variables(m: pyo.ConcreteModel, net) -> None:
    """Create ``m.s[l]`` (binary, closed = 1). Non-switchable branches are fixed at their
    initial status."""
    m.L = pyo.Set(initialize=net.branch_ids, ordered=True)
    m.B = pyo.Set(initialize=net.bus_ids, ordered=True)
    m.s = pyo.Var(m.L, domain=pyo.Binary)
    for l in net.branch_ids:
        if not net.switchable[l]:
            m.s[l].fix(net.s0[l])


def add_radiality_constraints(m: pyo.ConcreteModel, net, method: str = "scf") -> None:
    """Add spanning-tree constraints on ``m.s`` (see module docstring)."""
    if method not in RADIALITY_METHODS:
        raise ValueError(f"Unknown radiality method '{method}'. Choose from {RADIALITY_METHODS}.")
    n = len(net.bus_ids)
    big = n - 1
    root = net.root_bus
    m.rad_fwd = pyo.Var(m.L, bounds=(0, big), initialize=0.0)   # commodity on arc from->to  [units]
    m.rad_bwd = pyo.Var(m.L, bounds=(0, big), initialize=0.0)   # commodity on arc to->from  [units]
    m.rad_count = pyo.Constraint(expr=sum(m.s[l] for l in net.branch_ids) == n - 1)

    if method == "scf":
        m.rad_cap = pyo.Constraint(m.L, rule=lambda m, l: m.rad_fwd[l] + m.rad_bwd[l] <= big * m.s[l])
    else:
        m.rad_y_fwd = pyo.Var(m.L, domain=pyo.Binary)
        m.rad_y_bwd = pyo.Var(m.L, domain=pyo.Binary)
        m.rad_link = pyo.Constraint(m.L, rule=lambda m, l: m.s[l] == m.rad_y_fwd[l] + m.rad_y_bwd[l])
        m.rad_cap_fwd = pyo.Constraint(m.L, rule=lambda m, l: m.rad_fwd[l] <= big * m.rad_y_fwd[l])
        m.rad_cap_bwd = pyo.Constraint(m.L, rule=lambda m, l: m.rad_bwd[l] <= big * m.rad_y_bwd[l])

        def parent_rule(m, i):
            incoming = sum(m.rad_y_fwd[l] for l in net.lines_to[i]) + \
                sum(m.rad_y_bwd[l] for l in net.lines_from[i])
            return incoming == (0 if i == root else 1)
        m.rad_parent = pyo.Constraint(m.B, rule=parent_rule)

    def flow_rule(m, i):
        out_ = sum(m.rad_fwd[l] for l in net.lines_from[i]) + sum(m.rad_bwd[l] for l in net.lines_to[i])
        in_ = sum(m.rad_bwd[l] for l in net.lines_from[i]) + sum(m.rad_fwd[l] for l in net.lines_to[i])
        return out_ - in_ == (big if i == root else -1)
    m.rad_flow = pyo.Constraint(m.B, rule=flow_rule)


def closing_expr(net, s: Dict[int, Any], l: int):
    """Closing operation indicator of branch l: s_l if initially open else 0."""
    return s[l] if net.s0[l] == 0 else 0


def switching_ops_expr(net, s: Dict[int, Any]):
    """Number of switching operations = sum_l |s_l - s0_l| (linear for binaries)."""
    return sum((1 - s[l]) if net.s0[l] == 1 else s[l] for l in net.branch_ids)


def switching_cost_expr(net, s: Dict[int, Any]):
    """First-stage switching cost [$] = sum_l c_l |s_l - s0_l|."""
    return sum(net.switch_cost[l] * ((1 - s[l]) if net.s0[l] == 1 else s[l]) for l in net.branch_ids)


def add_switching_limits(m: pyo.ConcreteModel, net, max_switch_ops=None,
                         exclusive_close_pairs: Iterable[Tuple[int, int]] = ()) -> None:
    """Optional first-stage limits.

    * ``max_switch_ops``: at most this many switch operations (open + close) w.r.t. the initial state.
    * ``exclusive_close_pairs``: pairs (a, b) of branches that may not both be *closed by an
      operation* (i.e. not both switched from open to closed).
    """
    if max_switch_ops is not None:
        m.max_ops = pyo.Constraint(expr=switching_ops_expr(net, m.s) <= max_switch_ops)
    pairs = list(exclusive_close_pairs)
    if pairs:
        m.excl_close = pyo.ConstraintList()
        for a, b in pairs:
            expr = closing_expr(net, m.s, a) + closing_expr(net, m.s, b)
            if not isinstance(expr, (int, float)):
                m.excl_close.add(expr <= 1)
