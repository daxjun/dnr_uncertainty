"""Nonlinear AC power flow (backward/forward sweep) for post-optimisation validation ONLY.

The MILP uses LinDistFlow; this exact radial AC solution (per unit, constant-power loads)
quantifies the linearisation error in voltage, current and losses.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import networkx as nx


def ac_power_flow(net, closed_branches: Iterable[int], p_net_pu: Dict[int, float],
                  q_net_pu: Dict[int, float], tol: float = 1e-10, max_iter: int = 200) -> Dict:
    """Backward/forward sweep on the radial network.

    ``p_net_pu`` / ``q_net_pu`` are net *consumptions* per bus [p.u.] (load - shed - PV used).
    Returns dict with ``v`` (|V| p.u.), ``i_pu`` (branch current magnitude p.u., key=branch id),
    ``loss_pu``, ``converged``, ``iterations``.
    """
    closed = list(closed_branches)
    g = nx.Graph()
    for l in closed:
        g.add_edge(net.from_bus[l], net.to_bus[l], l=l)
    root = net.root_bus
    order = list(nx.bfs_tree(g, root))
    par: Dict[int, Tuple[int, int]] = {}
    for u, w in nx.bfs_edges(g, root):
        par[w] = (u, g[u][w]["l"])
    V = {b: complex(net.v_root_pu, 0.0) for b in order}
    I_br: Dict[int, complex] = {}
    it, converged = 0, False
    for it in range(1, max_iter + 1):
        s_load = {b: complex(p_net_pu.get(b, 0.0), q_net_pu.get(b, 0.0)) for b in order}
        i_inj = {b: (s_load[b] / V[b]).conjugate() for b in order}
        cur = dict(i_inj)
        for b in reversed(order):
            if b == root:
                continue
            u, l = par[b]
            I_br[l] = cur[b]
            cur[u] = cur[u] + cur[b]
        newV = {root: V[root]}
        for b in order:
            if b == root:
                continue
            u, l = par[b]
            z = complex(net.r_pu[l], net.x_pu[l])
            newV[b] = newV[u] - z * I_br[l]
        err = max(abs(newV[b] - V[b]) for b in order)
        V = newV
        if err < tol:
            converged = True
            break
    loss = sum(net.r_pu[l] * abs(I_br[l]) ** 2 for l in I_br)
    return dict(v={b: abs(V[b]) for b in order}, i_pu={l: abs(c) for l, c in I_br.items()},
                loss_pu=loss, converged=converged, iterations=it)
