"""Matplotlib / networkx figures. All functions save a PNG and return its path."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_voltage_profiles(net, sol: Dict[str, Any], path: Path, title: str = "Bus voltage profile") -> Path:
    """Voltage magnitude [p.u.] vs bus for every scenario, with limits."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for sid, d in sol["scenarios"].items():
        buses = sorted(d["voltage_pu"])
        ax.plot(buses, [d["voltage_pu"][b] for b in buses], marker=".", lw=1, label=sid)
    ax.axhline(min(net.vmin.values()), color="r", ls="--", lw=1)
    ax.axhline(max(net.vmax.values()), color="r", ls="--", lw=1)
    ax.set_xlabel("Bus ID")
    ax.set_ylabel("|V| [p.u.]")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_line_loading(net, sol: Dict[str, Any], path: Path, title: str = "Line loading") -> Path:
    """Branch current as % of limit for every scenario (closed branches only)."""
    closed = sol["closed_branches"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    scen = list(sol["scenarios"].items())
    w = 0.8 / max(len(scen), 1)
    for k, (sid, d) in enumerate(scen):
        ax.bar([i + k * w for i in range(len(closed))], [d["loading_pct"][l] for l in closed], width=w, label=sid)
    ax.set_xticks([i + 0.4 for i in range(len(closed))])
    ax.set_xticklabels(closed, fontsize=7)
    ax.axhline(100, color="r", ls="--", lw=1)
    ax.set_xlabel("Closed branch ID")
    ax.set_ylabel("Loading [% of I_max]")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)


def plot_pv(sol: Dict[str, Any], path: Path, title: str = "PV generation and curtailment") -> Path:
    """Available PV split into used and curtailed energy per scenario [kW]."""
    ids = list(sol["scenarios"])
    avail = [sum(sol["scenarios"][s]["pv_available_kw"].values()) for s in ids]
    curt = [sol["scenarios"][s]["curtailment_total_kw"] for s in ids]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ids, [a - c for a, c in zip(avail, curt)], label="PV used")
    ax.bar(ids, curt, bottom=[a - c for a, c in zip(avail, curt)], color="tab:red", label="Curtailed")
    ax.set_ylabel("Power [kW]")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    ax.legend()
    return _save(fig, path)


def plot_load(sol: Dict[str, Any], path: Path, title: str = "Load served and shed") -> Path:
    """Served and shed active load per scenario [kW]."""
    ids = list(sol["scenarios"])
    tot = [sol["scenarios"][s]["load_total_kw"] for s in ids]
    shed = [sol["scenarios"][s]["load_shedding_total_kw"] for s in ids]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ids, [t - s for t, s in zip(tot, shed)], label="Served")
    ax.bar(ids, shed, bottom=[t - s for t, s in zip(tot, shed)], color="tab:red", label="Shed")
    ax.set_ylabel("Power [kW]")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    ax.legend()
    return _save(fig, path)


def plot_topology(net, sol: Dict[str, Any], path: Path, title: str = "Final radial topology") -> Path:
    """Network diagram: closed branches solid, opened (candidate) branches dashed."""
    g = nx.Graph()
    g.add_nodes_from(net.bus_ids)
    for l in net.branch_ids:
        g.add_edge(net.from_bus[l], net.to_bus[l], l=l)
    pos = nx.kamada_kawai_layout(g)
    closed = set(sol["closed_branches"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ce = [(net.from_bus[l], net.to_bus[l]) for l in net.branch_ids if l in closed]
    oe = [(net.from_bus[l], net.to_bus[l]) for l in net.branch_ids if l not in closed]
    nx.draw_networkx_edges(g, pos, edgelist=ce, width=2, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=oe, style="dashed", edge_color="tab:red", width=1.2, ax=ax)
    colors = ["gold" if b == net.root_bus else ("tab:green" if b in net.pv_buses else "lightsteelblue")
              for b in net.bus_ids]
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=260, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=7, ax=ax)
    lab = {(net.from_bus[l], net.to_bus[l]): str(l) for l in net.branch_ids if l not in closed}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=lab, font_size=6, font_color="red", ax=ax)
    ax.set_title(f"{title}  (red dashed = open; gold = substation; green = PV bus)")
    ax.axis("off")
    return _save(fig, path)


def plot_ccg_convergence(history: Sequence[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Lower/upper bound per C&CG iteration."""
    if not history:
        return None
    it = [h["iteration"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(it, [h["lower_bound"] for h in history], marker="o", label="Lower bound (master)")
    ax.plot(it, [h["upper_bound"] for h in history], marker="s", label="Cost vs worst found")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost [$]")
    ax.legend()
    ax.set_title("C&CG convergence")
    return _save(fig, path)


def make_all_plots(net, sol: Dict[str, Any], fig_dir: Path, prefix: str) -> list:
    """Create the five required figures; returns their paths."""
    fig_dir = Path(fig_dir)
    return [plot_voltage_profiles(net, sol, fig_dir / f"{prefix}_voltage_profile.png"),
            plot_line_loading(net, sol, fig_dir / f"{prefix}_line_loading.png"),
            plot_pv(sol, fig_dir / f"{prefix}_pv_curtailment.png"),
            plot_load(sol, fig_dir / f"{prefix}_load_shedding.png"),
            plot_topology(net, sol, fig_dir / f"{prefix}_topology.png")]
