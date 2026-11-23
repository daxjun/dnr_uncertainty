"""LinDistFlow operation block (second stage) for ONE operating scenario / realisation.

Sign convention and units
-------------------------
Per unit on (S_base, V_base).  Branch l = (i -> j) has orientation from_bus -> to_bus.
``P[l], Q[l]``  active/reactive power flowing from ``from_bus`` into the branch   [p.u.]
Bus power balance (non-root bus i, "net consumption" convention, lossless LinDistFlow):

    sum_{l: to=i} P_l - sum_{l: from=i} P_l = p_load_i - p_shed_i - (pv_i - p_curt_i)
    sum_{l: to=i} Q_l - sum_{l: from=i} Q_l = q_load_i - tan(phi_i) p_shed_i

PV is a negative load at unity power factor.  ``pv_i`` is the *available* PV power, ``p_curt_i``
the curtailed part (0 <= p_curt <= pv), ``p_pv_used = pv - p_curt``.  Load shedding removes
active load and, at constant power factor, reactive load.

Voltage drop (squared voltage v2 = |V|^2), closed branch:   v2_i - v2_j = 2 (r P + x Q)
Open branch: relaxed by big-M (``M_l (1 - s_l)``) and P = Q = 0 (via ``|P|,|Q| <= S s_l``).

Approximations (all documented in the README):
  1. Losses neglected in the balance and in the voltage drop (LinDistFlow).
  2. I^2 ~ P^2 + Q^2 (voltage taken as 1 p.u. in the current/loss terms).
  3. P^2 and Q^2 are replaced by their convex piecewise-linear *tangent* lower envelopes
     (variables ``P2``, ``Q2``): a lower approximation, exact at the breakpoints.
     I2_l = P2_l + Q2_l ;  loss_l = r_l * I2_l.
  4. Thermal limit |S| <= S_max (at 1 p.u.) is replaced by an inscribed octagon
     (|P|,|Q| <= a s, |P|+|Q| <= sqrt(2) a s with a = S_max cos(pi/8)): conservative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pyomo.environ as pyo


@dataclass
class DNRConfig:
    """Configuration shared by the stochastic and robust models.

    Costs: ``loss_cost_per_mwh`` [$/MWh], ``curtailment_cost_per_mwh`` [$/MWh],
    ``shedding_cost_per_mwh`` [$/MWh], ``voltage_violation_cost`` [$ per p.u. of squared-voltage
    violation per bus-hour] (only with ``soft_voltage``).  Operating costs of a scenario are
    energy prices x power x ``duration_h`` (scenario value, else ``default_hours``).
    """

    solver: str = "appsi_highs"
    solver_options: Dict[str, Any] = field(default_factory=dict)
    time_limit_s: float = 300.0
    mip_gap: float = 1e-4
    tee: bool = False
    radiality_method: str = "scf"          # "scf" | "directed"
    loss_cost_per_mwh: float = 60.0
    curtailment_cost_per_mwh: float = 120.0
    shedding_cost_per_mwh: float = 5000.0
    voltage_violation_cost: float = 1.0e5
    default_hours: float = 24.0
    use_switching_cost: bool = True
    max_switch_ops: Optional[int] = None
    exclusive_close_pairs: Sequence[Tuple[int, int]] = ()
    soft_voltage: bool = False
    allow_shedding: bool = True
    allow_curtailment: bool = True
    # breakpoints of the P^2/Q^2 tangent envelope as fractions of the (inscribed) flow limit
    loss_breakpoints: Sequence[float] = (0.0, 0.06, 0.15, 0.30, 0.55, 1.0)
    soft_v2_margin: float = 0.25           # max |v2| violation when soft_voltage [p.u.^2]


def _is_const(x: Any) -> bool:
    return isinstance(x, (int, float))


def build_operation_block(blk: pyo.Block, net, s: Mapping[int, Any], p_load: Mapping[int, Any],
                          q_load: Mapping[int, Any], tanphi: Mapping[int, float],
                          pv: Mapping[int, Any], cfg: DNRConfig, hours: float) -> None:
    """Populate ``blk`` with the second-stage LinDistFlow model.

    Parameters
    ----------
    s        : branch_id -> Pyomo Var (master) or number 0/1 (fixed topology -> pure LP).
    p_load   : bus_id -> active load [p.u.]   (number or linear expression)
    q_load   : bus_id -> reactive load [p.u.] (number or linear expression)
    tanphi   : bus_id -> tan(phi) used to shed reactive power with active power
    pv       : bus_id -> available PV power [p.u.] (number or linear expression)
    hours    : duration represented by this block [h] (cost multiplier)

    Creates variables P, Q, P2, Q2, v2, p_shed, p_curt (and vlo/vhi if soft voltage) and the
    expressions ``loss_pu`` [p.u.], ``curt_pu``, ``shed_pu`` and ``cost`` [$].
    """
    root = net.root_bus
    B, Ls = net.bus_ids, net.branch_ids
    margin = cfg.soft_v2_margin if cfg.soft_voltage else 0.0

    blk.P = pyo.Var(Ls, domain=pyo.Reals, initialize=0.0)      # active flow  [p.u.]
    blk.Q = pyo.Var(Ls, domain=pyo.Reals, initialize=0.0)      # reactive flow [p.u.]
    blk.P2 = pyo.Var(Ls, domain=pyo.NonNegativeReals, initialize=0.0)  # >= P^2 envelope [p.u.^2]
    blk.Q2 = pyo.Var(Ls, domain=pyo.NonNegativeReals, initialize=0.0)
    blk.v2 = pyo.Var(B, domain=pyo.Reals, initialize=1.0,
                     bounds=lambda b, i: (net.vmin[i] ** 2 - margin, net.vmax[i] ** 2 + margin))
    blk.v2[root].fix(net.v_root_pu ** 2)

    blk.shed_buses = [i for i in net.load_buses]
    blk.pv_buses = [i for i in net.pv_buses]
    blk.p_shed = pyo.Var(blk.shed_buses, domain=pyo.NonNegativeReals, initialize=0.0)   # [p.u.]
    blk.p_curt = pyo.Var(blk.pv_buses, domain=pyo.NonNegativeReals, initialize=0.0)     # [p.u.]
    if not cfg.allow_shedding:
        for i in blk.shed_buses:
            blk.p_shed[i].fix(0.0)
    if not cfg.allow_curtailment:
        for i in blk.pv_buses:
            blk.p_curt[i].fix(0.0)
    if cfg.soft_voltage:
        blk.vlo = pyo.Var(B, bounds=(0, margin), initialize=0.0)
        blk.vhi = pyo.Var(B, bounds=(0, margin), initialize=0.0)
        blk.vlo[root].fix(0.0)
        blk.vhi[root].fix(0.0)

    blk.lim = pyo.ConstraintList()       # thermal octagon / flow-on-closed-branch limits
    blk.vdrop = pyo.ConstraintList()     # voltage drop
    blk.env = pyo.ConstraintList()       # P^2, Q^2 tangent envelopes
    blk.bal_p = pyo.ConstraintList()
    blk.bal_q = pyo.ConstraintList()
    blk.dev = pyo.ConstraintList()       # shed/curtail upper bounds, soft-voltage links

    r2 = math.sqrt(2.0)
    for l in Ls:
        sl = s[l]
        if _is_const(sl) and sl < 0.5:      # branch fixed open: no flow
            for v in (blk.P[l], blk.Q[l], blk.P2[l], blk.Q2[l]):
                v.fix(0.0)
            continue
        a = net.s_eff_pu[l]
        blk.lim.add(blk.P[l] <= a * sl)
        blk.lim.add(blk.P[l] >= -a * sl)
        blk.lim.add(blk.Q[l] <= a * sl)
        blk.lim.add(blk.Q[l] >= -a * sl)
        blk.lim.add(blk.P[l] + blk.Q[l] <= r2 * a * sl)
        blk.lim.add(blk.P[l] - blk.Q[l] <= r2 * a * sl)
        blk.lim.add(-blk.P[l] + blk.Q[l] <= r2 * a * sl)
        blk.lim.add(-blk.P[l] - blk.Q[l] <= r2 * a * sl)
        # tangent envelope of P^2, Q^2:  X2 >= 2 c X - c^2  for breakpoints c
        levels = sorted({sgn * f * a for f in cfg.loss_breakpoints for sgn in (-1.0, 1.0)})
        for c in levels:
            blk.env.add(blk.P2[l] >= 2.0 * c * blk.P[l] - c * c)
            blk.env.add(blk.Q2[l] >= 2.0 * c * blk.Q[l] - c * c)
        i, j = net.from_bus[l], net.to_bus[l]
        drop = blk.v2[i] - blk.v2[j] - 2.0 * (net.r_pu[l] * blk.P[l] + net.x_pu[l] * blk.Q[l])
        if _is_const(sl):                  # closed
            blk.vdrop.add(drop == 0.0)
        else:                              # big-M switching of the voltage-drop equation
            bigm = (net.vmax[i] ** 2 - net.vmin[j] ** 2 + 2 * margin) + \
                   2.0 * (net.r_pu[l] + net.x_pu[l]) * a * r2 + 0.05
            blk.vdrop.add(drop <= bigm * (1 - sl))
            blk.vdrop.add(drop >= -bigm * (1 - sl))

    for i in B:
        if i == root:
            continue
        if cfg.soft_voltage:
            blk.dev.add(blk.v2[i] >= net.vmin[i] ** 2 - blk.vlo[i])
            blk.dev.add(blk.v2[i] <= net.vmax[i] ** 2 + blk.vhi[i])
        inflow_p = sum(blk.P[l] for l in net.lines_to[i]) - sum(blk.P[l] for l in net.lines_from[i])
        inflow_q = sum(blk.Q[l] for l in net.lines_to[i]) - sum(blk.Q[l] for l in net.lines_from[i])
        dem_p = p_load.get(i, 0.0)
        dem_q = q_load.get(i, 0.0)
        if i in blk.shed_buses:
            dem_p = dem_p - blk.p_shed[i]
            dem_q = dem_q - tanphi.get(i, 0.0) * blk.p_shed[i]
            blk.dev.add(blk.p_shed[i] <= p_load.get(i, 0.0))
        if i in blk.pv_buses:
            dem_p = dem_p - (pv.get(i, 0.0) - blk.p_curt[i])
            blk.dev.add(blk.p_curt[i] <= pv.get(i, 0.0))
        blk.bal_p.add(inflow_p == dem_p)
        blk.bal_q.add(inflow_q == dem_q)

    sbase = net.s_base_mva
    blk.loss_pu = pyo.Expression(expr=pyo.quicksum(net.r_pu[l] * (blk.P2[l] + blk.Q2[l]) for l in Ls))
    blk.curt_pu = pyo.Expression(expr=pyo.quicksum(blk.p_curt[i] for i in blk.pv_buses))
    blk.shed_pu = pyo.Expression(expr=pyo.quicksum(blk.p_shed[i] for i in blk.shed_buses))
    viol = pyo.quicksum(blk.vlo[i] + blk.vhi[i] for i in B) if cfg.soft_voltage else 0.0
    blk.viol_pu = pyo.Expression(expr=viol)
    blk.cost = pyo.Expression(expr=hours * (
        sbase * (cfg.loss_cost_per_mwh * blk.loss_pu + cfg.curtailment_cost_per_mwh * blk.curt_pu
                 + cfg.shedding_cost_per_mwh * blk.shed_pu) + cfg.voltage_violation_cost * blk.viol_pu))
