"""Network data structures, per-unit conversion, CSV/Excel I/O and the IEEE 33-bus feeder.

Units used throughout the project
---------------------------------
* Input tables : kW, kVAr, ohm, A, kV  (engineering units)
* Optimisation : per unit (p.u.) on ``s_base_mva`` / ``v_base_kv``
* Reporting    : kW, kVAr, p.u. voltage, A

IEEE 33-bus data source
-----------------------
M. E. Baran and F. F. Wu, "Network reconfiguration in distribution systems for loss
reduction and load balancing," IEEE Trans. Power Delivery, vol. 4, no. 2, pp. 1401-1407,
Apr. 1989.  Base: 12.66 kV, 10 MVA (Z_base = 16.03 ohm), total load 3715 kW + j2300 kVAr,
37 branches (32 sectionalising + 5 normally-open tie switches, branches 33-37).

NOT from the paper (assumptions of this project): branch ampacities, PV placement/size,
load-profile assignment, switching costs and the substation voltage set-point.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
EXAMPLE_BRANCHES_CSV = DATA_DIR / "example_network.csv"
EXAMPLE_BUSES_CSV = DATA_DIR / "example_buses.csv"
EXAMPLE_META_JSON = DATA_DIR / "example_meta.json"

# (branch_id, from, to, R [ohm], X [ohm]) -- Baran & Wu (1989); 33-37 are tie switches.
_IEEE33_BRANCHES = [
    (1, 1, 2, 0.0922, 0.0470), (2, 2, 3, 0.4930, 0.2511), (3, 3, 4, 0.3660, 0.1864),
    (4, 4, 5, 0.3811, 0.1941), (5, 5, 6, 0.8190, 0.7070), (6, 6, 7, 0.1872, 0.6188),
    (7, 7, 8, 0.7114, 0.2351), (8, 8, 9, 1.0300, 0.7400), (9, 9, 10, 1.0440, 0.7400),
    (10, 10, 11, 0.1966, 0.0650), (11, 11, 12, 0.3744, 0.1238), (12, 12, 13, 1.4680, 1.1550),
    (13, 13, 14, 0.5416, 0.7129), (14, 14, 15, 0.5910, 0.5260), (15, 15, 16, 0.7463, 0.5450),
    (16, 16, 17, 1.2890, 1.7210), (17, 17, 18, 0.7320, 0.5740), (18, 2, 19, 0.1640, 0.1565),
    (19, 19, 20, 1.5042, 1.3554), (20, 20, 21, 0.4095, 0.4784), (21, 21, 22, 0.7089, 0.9373),
    (22, 3, 23, 0.4512, 0.3083), (23, 23, 24, 0.8980, 0.7091), (24, 24, 25, 0.8960, 0.7011),
    (25, 6, 26, 0.2030, 0.1034), (26, 26, 27, 0.2842, 0.1447), (27, 27, 28, 1.0590, 0.9337),
    (28, 28, 29, 0.8042, 0.7006), (29, 29, 30, 0.5075, 0.2585), (30, 30, 31, 0.9744, 0.9630),
    (31, 31, 32, 0.3105, 0.3619), (32, 32, 33, 0.3410, 0.5302),
    (33, 8, 21, 2.0, 2.0), (34, 9, 15, 2.0, 2.0), (35, 12, 22, 2.0, 2.0),
    (36, 18, 33, 0.5, 0.5), (37, 25, 29, 0.5, 0.5),
]
# bus: (P [kW], Q [kVAr]) -- Baran & Wu (1989)
_IEEE33_LOADS = {
    2: (100, 60), 3: (90, 40), 4: (120, 80), 5: (60, 30), 6: (60, 20), 7: (200, 100),
    8: (200, 100), 9: (60, 20), 10: (60, 20), 11: (45, 30), 12: (60, 35), 13: (60, 35),
    14: (120, 80), 15: (60, 10), 16: (60, 20), 17: (60, 20), 18: (90, 40), 19: (90, 40),
    20: (90, 40), 21: (90, 40), 22: (90, 40), 23: (90, 50), 24: (420, 200), 25: (420, 200),
    26: (60, 25), 27: (60, 25), 28: (60, 20), 29: (120, 70), 30: (200, 600), 31: (150, 70),
    32: (210, 100), 33: (60, 40),
}
# ASSUMED (not in the paper): PV plants at feeder-end buses.
_IEEE33_PV_KW = {18: 500.0, 25: 800.0, 33: 600.0}
_COMMERCIAL_BUSES = {24, 25, 29, 30, 31, 32}
_HIGH_AMPACITY_BRANCHES = {1, 2, 3, 4, 5, 18, 22, 25}  # trunk sections: 400 A, others 250 A (assumed)


def ieee33_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (buses, branches, meta) for the IEEE 33-bus feeder (engineering units)."""
    bus_rows = []
    for b in range(1, 34):
        p, q = _IEEE33_LOADS.get(b, (0.0, 0.0))
        pv = _IEEE33_PV_KW.get(b, 0.0)
        btype = "root" if b == 1 else ("pv_load" if pv > 0 else "load")
        bus_rows.append(dict(
            bus_id=b, bus_type=btype, p_load_kw=float(p), q_load_kvar=float(q),
            v_min_pu=0.95, v_max_pu=1.05, pv_capacity_kw=pv,
            load_profile="commercial" if b in _COMMERCIAL_BUSES else "residential",
            pv_profile="pv_clear_sky" if pv > 0 else "",
        ))
    br_rows = []
    for bid, f, t, r, x in _IEEE33_BRANCHES:
        tie = bid >= 33
        br_rows.append(dict(
            branch_id=bid, from_bus=f, to_bus=t, r_ohm=r, x_ohm=x,
            i_max_a=400.0 if bid in _HIGH_AMPACITY_BRANCHES else 250.0,
            initial_status=0 if tie else 1, normally_open=1 if tie else 0,
            switchable=1, switch_cost=5.0,
        ))
    meta = dict(name="IEEE 33-bus (Baran & Wu 1989)", s_base_mva=10.0, v_base_kv=12.66,
                root_bus=1, v_root_pu=1.03)
    return pd.DataFrame(bus_rows), pd.DataFrame(br_rows), meta


def write_example_files(directory: Union[str, Path] = DATA_DIR, overwrite: bool = False) -> None:
    """Write the IEEE 33-bus example as CSV (+ JSON metadata) into ``directory``."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    buses, branches, meta = ieee33_dataframes()
    targets = [(d / "example_buses.csv", buses), (d / "example_network.csv", branches)]
    for path, df in targets:
        if overwrite or not path.exists():
            df.to_csv(path, index=False)
    meta_path = d / "example_meta.json"
    if overwrite or not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2))


@dataclass
class Network:
    """A distribution feeder with candidate (switchable) branches.

    Attributes
    ----------
    buses : DataFrame  [bus_id, bus_type, p_load_kw, q_load_kvar, v_min_pu, v_max_pu,
                        pv_capacity_kw, load_profile, pv_profile]
    branches : DataFrame [branch_id, from_bus, to_bus, r_ohm, x_ohm, i_max_a, initial_status,
                          normally_open, switchable, switch_cost]
    s_base_mva, v_base_kv : per-unit bases.  v_root_pu : substation voltage set-point [p.u.].

    Derived per-unit dictionaries (all keyed by bus_id or branch_id) are built in
    ``__post_init__``: ``r_pu, x_pu, s_max_pu, s_eff_pu, vmin, vmax, p_load_pu, q_load_pu,
    pv_cap_pu, lines_from, lines_to``.  ``s_eff_pu`` is the thermal limit shrunk by cos(pi/8)
    so that the linear octagon used in the MILP is *inscribed* in the apparent-power circle.
    """

    buses: pd.DataFrame
    branches: pd.DataFrame
    s_base_mva: float = 10.0
    v_base_kv: float = 12.66
    root_bus: int = 1
    v_root_pu: float = 1.0
    name: str = "network"

    bus_ids: List[int] = field(init=False, default_factory=list)
    branch_ids: List[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        b = self.buses.copy()
        br = self.branches.copy()
        defaults_b = dict(p_load_kw=0.0, q_load_kvar=0.0, v_min_pu=0.95, v_max_pu=1.05,
                          pv_capacity_kw=0.0, load_profile="residential", pv_profile="")
        for c, v in defaults_b.items():
            if c not in b.columns:
                b[c] = v
        b["pv_profile"] = b["pv_profile"].fillna("")
        if "bus_type" not in b.columns:
            b["bus_type"] = "load"
        if "initial_status" not in br.columns:
            br["initial_status"] = 1
        if "normally_open" not in br.columns:
            br["normally_open"] = 1 - br["initial_status"]
        if "switchable" not in br.columns:
            br["switchable"] = 1
        if "switch_cost" not in br.columns:
            br["switch_cost"] = 0.0
        if "i_max_a" not in br.columns:
            br["i_max_a"] = 400.0
        b["bus_id"] = b["bus_id"].astype(int)
        for c in ("branch_id", "from_bus", "to_bus"):
            br[c] = br[c].astype(int)
        self.buses, self.branches = b.reset_index(drop=True), br.reset_index(drop=True)
        self.bus_ids = [int(i) for i in self.buses["bus_id"]]
        self.branch_ids = [int(i) for i in self.branches["branch_id"]]
        self._build_derived()
        self.validate()

    # ------------------------------------------------------------------ bases
    @property
    def z_base_ohm(self) -> float:
        """Impedance base [ohm] = V_base^2 / S_base."""
        return self.v_base_kv ** 2 / self.s_base_mva

    @property
    def i_base_a(self) -> float:
        """Current base [A] = S_base / (sqrt(3) V_base)."""
        return self.s_base_mva * 1000.0 / (math.sqrt(3.0) * self.v_base_kv)

    def kw_to_pu(self, kw: float) -> float:
        """Convert kW (or kVAr) to p.u. on s_base_mva."""
        return kw / 1000.0 / self.s_base_mva

    def pu_to_kw(self, pu: float) -> float:
        """Convert p.u. power to kW (or kVAr)."""
        return pu * self.s_base_mva * 1000.0

    # --------------------------------------------------------------- internals
    def _build_derived(self) -> None:
        zb = self.z_base_ohm
        bi = self.buses.set_index("bus_id")
        br = self.branches.set_index("branch_id")
        self.from_bus = {l: int(br.at[l, "from_bus"]) for l in self.branch_ids}
        self.to_bus = {l: int(br.at[l, "to_bus"]) for l in self.branch_ids}
        self.r_pu = {l: float(br.at[l, "r_ohm"]) / zb for l in self.branch_ids}
        self.x_pu = {l: float(br.at[l, "x_ohm"]) / zb for l in self.branch_ids}
        self.i_max_a = {l: float(br.at[l, "i_max_a"]) for l in self.branch_ids}
        # |S| limit at 1 p.u. voltage: sqrt(3) V_base I_max / S_base
        self.s_max_pu = {l: math.sqrt(3.0) * self.v_base_kv * self.i_max_a[l] / 1000.0 / self.s_base_mva
                         for l in self.branch_ids}
        self.s_eff_pu = {l: self.s_max_pu[l] * math.cos(math.pi / 8.0) for l in self.branch_ids}
        self.s0 = {l: int(br.at[l, "initial_status"]) for l in self.branch_ids}
        self.normally_open = {l: int(br.at[l, "normally_open"]) for l in self.branch_ids}
        self.switchable = {l: int(br.at[l, "switchable"]) for l in self.branch_ids}
        self.switch_cost = {l: float(br.at[l, "switch_cost"]) for l in self.branch_ids}
        self.vmin = {i: float(bi.at[i, "v_min_pu"]) for i in self.bus_ids}
        self.vmax = {i: float(bi.at[i, "v_max_pu"]) for i in self.bus_ids}
        self.p_load_pu = {i: self.kw_to_pu(float(bi.at[i, "p_load_kw"])) for i in self.bus_ids}
        self.q_load_pu = {i: self.kw_to_pu(float(bi.at[i, "q_load_kvar"])) for i in self.bus_ids}
        self.pv_cap_pu = {i: self.kw_to_pu(float(bi.at[i, "pv_capacity_kw"])) for i in self.bus_ids}
        self.lines_from = {i: [l for l in self.branch_ids if self.from_bus[l] == i] for i in self.bus_ids}
        self.lines_to = {i: [l for l in self.branch_ids if self.to_bus[l] == i] for i in self.bus_ids}
        self.load_buses = [i for i in self.bus_ids if i != self.root_bus and self.p_load_pu[i] > 0]
        self.pv_buses = [i for i in self.bus_ids if i != self.root_bus and self.pv_cap_pu[i] > 0]

    def validate(self) -> None:
        """Raise ValueError if the data are inconsistent."""
        if len(set(self.bus_ids)) != len(self.bus_ids):
            raise ValueError("Duplicate bus IDs.")
        if len(set(self.branch_ids)) != len(self.branch_ids):
            raise ValueError("Duplicate branch IDs.")
        if self.root_bus not in self.bus_ids:
            raise ValueError(f"Root bus {self.root_bus} not in bus table.")
        for l in self.branch_ids:
            if self.from_bus[l] not in self.bus_ids or self.to_bus[l] not in self.bus_ids:
                raise ValueError(f"Branch {l} references an unknown bus.")
            if self.from_bus[l] == self.to_bus[l]:
                raise ValueError(f"Branch {l} is a self-loop.")
            if self.r_pu[l] < 0 or self.x_pu[l] < 0 or (self.r_pu[l] + self.x_pu[l]) <= 0:
                raise ValueError(f"Branch {l} has invalid impedance.")
            if self.i_max_a[l] <= 0:
                raise ValueError(f"Branch {l} has a non-positive current limit.")
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(self.bus_ids)
        g.add_edges_from((self.from_bus[l], self.to_bus[l]) for l in self.branch_ids)
        if not nx.is_connected(g):
            raise ValueError("The candidate branch set does not connect all buses to the root.")

    def initial_closed_branches(self) -> List[int]:
        """Branch IDs that are closed in the initial configuration."""
        return [l for l in self.branch_ids if self.s0[l] == 1]


# --------------------------------------------------------------------------- I/O
def _meta_defaults(buses: pd.DataFrame, meta: Optional[dict]) -> dict:
    out = dict(name="network", s_base_mva=10.0, v_base_kv=12.66, v_root_pu=1.0)
    if "bus_type" in buses.columns and (buses["bus_type"] == "root").any():
        out["root_bus"] = int(buses.loc[buses["bus_type"] == "root", "bus_id"].iloc[0])
    else:
        out["root_bus"] = int(buses["bus_id"].iloc[0])
    out.update(meta or {})
    return out


def load_network_csv(branches_csv: Union[str, Path], buses_csv: Union[str, Path],
                     meta_json: Optional[Union[str, Path]] = None) -> Network:
    """Load a Network from a branch CSV, a bus CSV and an optional JSON metadata file."""
    buses = pd.read_csv(buses_csv)
    branches = pd.read_csv(branches_csv)
    meta = json.loads(Path(meta_json).read_text()) if meta_json and Path(meta_json).exists() else None
    m = _meta_defaults(buses, meta)
    return Network(buses=buses, branches=branches, s_base_mva=float(m["s_base_mva"]),
                   v_base_kv=float(m["v_base_kv"]), root_bus=int(m["root_bus"]),
                   v_root_pu=float(m["v_root_pu"]), name=str(m["name"]))


def load_network_excel(path: Union[str, Path]) -> Network:
    """Load a Network from an Excel workbook with sheets ``buses``, ``branches`` and an
    optional ``meta`` sheet (two columns: key, value)."""
    sheets = pd.read_excel(path, sheet_name=None)
    meta = None
    if "meta" in sheets:
        meta = {str(k): v for k, v in zip(sheets["meta"].iloc[:, 0], sheets["meta"].iloc[:, 1])}
    m = _meta_defaults(sheets["buses"], meta)
    return Network(buses=sheets["buses"], branches=sheets["branches"],
                   s_base_mva=float(m["s_base_mva"]), v_base_kv=float(m["v_base_kv"]),
                   root_bus=int(m["root_bus"]), v_root_pu=float(m["v_root_pu"]), name=str(m["name"]))


def save_network_excel(net: Network, path: Union[str, Path]) -> None:
    """Write a Network to an Excel workbook readable by :func:`load_network_excel`."""
    meta = pd.DataFrame(dict(key=["name", "s_base_mva", "v_base_kv", "root_bus", "v_root_pu"],
                             value=[net.name, net.s_base_mva, net.v_base_kv, net.root_bus, net.v_root_pu]))
    with pd.ExcelWriter(path) as xw:
        net.buses.to_excel(xw, sheet_name="buses", index=False)
        net.branches.to_excel(xw, sheet_name="branches", index=False)
        meta.to_excel(xw, sheet_name="meta", index=False)


def load_example_network() -> Network:
    """Load the bundled IEEE 33-bus example (files are (re)created if missing)."""
    write_example_files(DATA_DIR, overwrite=False)
    return load_network_csv(EXAMPLE_BRANCHES_CSV, EXAMPLE_BUSES_CSV, EXAMPLE_META_JSON)
