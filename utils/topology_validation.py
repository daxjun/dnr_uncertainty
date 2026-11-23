"""Topology validation: connectivity, cycles, branch count, radial tree structure."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import networkx as nx


def validate_topology(net, closed_branches: Iterable[int]) -> Dict[str, Any]:
    """Check that the closed branches form a radial (spanning-tree) network rooted at the substation.

    Checks: every bus reachable from the root, no cycle (parallel branches count as a cycle),
    number of closed branches == N - 1, and tree structure.  Returns a dict with
    ``is_radial`` (overall) and the individual findings.
    """
    closed = list(closed_branches)
    n = len(net.bus_ids)
    g = nx.Graph()
    g.add_nodes_from(net.bus_ids)
    parent = {b: b for b in net.bus_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    independent_cycles = 0
    for l in closed:
        a, b = net.from_bus[l], net.to_bus[l]
        ra, rb = find(a), find(b)
        if ra == rb:
            independent_cycles += 1
        else:
            parent[ra] = rb
        g.add_edge(a, b)
    reachable = nx.node_connected_component(g, net.root_bus)
    unreachable = sorted(set(net.bus_ids) - reachable)
    connected = not unreachable
    has_cycle = independent_cycles > 0
    count_ok = len(closed) == n - 1
    depth = None
    if connected and not has_cycle:
        depth = max(nx.single_source_shortest_path_length(g, net.root_bus).values())
    is_radial = connected and not has_cycle and count_ok
    msgs: List[str] = []
    if not connected:
        msgs.append(f"{len(unreachable)} bus(es) not reachable from the substation: {unreachable}")
    if has_cycle:
        msgs.append(f"{independent_cycles} independent loop(s) present")
    if not count_ok:
        msgs.append(f"{len(closed)} closed branches, expected {n - 1}")
    return dict(is_radial=is_radial, is_connected=connected, has_cycle=has_cycle,
                n_independent_loops=independent_cycles, n_closed_branches=len(closed),
                expected_branches=n - 1, n_energized_buses=len(reachable),
                unreachable_buses=unreachable, tree_depth=depth,
                message="radial and connected" if is_radial else "; ".join(msgs))
