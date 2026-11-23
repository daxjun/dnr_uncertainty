"""Load profiles, the ``Scenario`` container and scenario CSV/Excel input/output.

Scenario table (long format, one row per scenario x bus):

    scenario_id, probability, period, duration_h, bus_id, p_load_kw, q_load_kvar, pv_kw

``probability`` and ``duration_h`` are constant within a scenario (``duration_h`` is optional,
blank/NaN = use the model default).  Units: kW, kVAr.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# Normalised hourly profiles (fraction of the peak value in the bus table), hours 0..23.
LOAD_PROFILES: Dict[str, List[float]] = {
    "residential": [0.52, 0.47, 0.44, 0.43, 0.45, 0.52, 0.64, 0.74, 0.76, 0.72, 0.70, 0.70,
                    0.70, 0.68, 0.68, 0.72, 0.82, 0.95, 1.00, 0.99, 0.96, 0.88, 0.75, 0.62],
    "commercial": [0.40, 0.38, 0.37, 0.37, 0.40, 0.48, 0.62, 0.78, 0.92, 0.98, 1.00, 1.00,
                   0.97, 1.00, 1.00, 0.98, 0.92, 0.82, 0.70, 0.62, 0.55, 0.50, 0.45, 0.42],
    "flat": [1.0] * 24,
}


@dataclass
class Scenario:
    """One operating scenario (second-stage realisation). Power in kW / kVAr.

    ``p_load_kw``, ``q_load_kvar`` and ``pv_kw`` (available PV *before* curtailment) are
    dictionaries bus_id -> value.  ``probability`` is the scenario weight (stochastic model) or
    the weight of the forecast period (robust model).
    """

    scenario_id: str
    probability: float
    p_load_kw: Dict[int, float]
    q_load_kvar: Dict[int, float]
    pv_kw: Dict[int, float]
    period: Optional[int] = None
    duration_h: Optional[float] = None
    meta: Dict[str, object] = field(default_factory=dict)


def load_forecast_kw(net, hour: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Forecast active/reactive load [kW, kVAr] per bus for a given hour (0-23).

    Peak values come from the bus table; the bus's ``load_profile`` gives the hourly factor.
    Reactive power follows the same factor (constant power factor).
    """
    p, q = {}, {}
    for _, row in net.buses.iterrows():
        bid = int(row["bus_id"])
        prof = LOAD_PROFILES.get(str(row["load_profile"]), LOAD_PROFILES["residential"])
        f = prof[hour % len(prof)]
        p[bid] = float(row["p_load_kw"]) * f
        q[bid] = float(row["q_load_kvar"]) * f
    return p, q


def normalize_probabilities(scenarios: Sequence[Scenario]) -> List[Scenario]:
    """Return scenarios with probabilities rescaled to sum to one."""
    tot = sum(s.probability for s in scenarios)
    if tot <= 0:
        raise ValueError("Scenario probabilities must have a positive sum.")
    for s in scenarios:
        s.probability = s.probability / tot
    return list(scenarios)


def validate_scenarios(net, scenarios: Sequence[Scenario], tol: float = 1e-6) -> None:
    """Raise ValueError if probabilities do not sum to one or values are negative/unknown."""
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if abs(sum(s.probability for s in scenarios) - 1.0) > 1e-6:
        raise ValueError("Scenario probabilities must sum to 1 (use normalize_probabilities).")
    ids = [s.scenario_id for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate scenario IDs.")
    for s in scenarios:
        if s.probability < 0:
            raise ValueError(f"Negative probability in scenario {s.scenario_id}.")
        for d, nm in ((s.p_load_kw, "p_load_kw"), (s.q_load_kvar, "q_load_kvar"), (s.pv_kw, "pv_kw")):
            for b, v in d.items():
                if b not in net.bus_ids:
                    raise ValueError(f"Scenario {s.scenario_id}: unknown bus {b} in {nm}.")
                if nm != "q_load_kvar" and v < -tol:
                    raise ValueError(f"Scenario {s.scenario_id}: negative {nm} at bus {b}.")
        for b, v in s.pv_kw.items():
            if v > net.pv_cap_pu[b] * net.s_base_mva * 1000.0 * (1 + 1e-6) + 1e-6:
                raise ValueError(f"Scenario {s.scenario_id}: PV at bus {b} exceeds installed capacity.")


def scenarios_to_dataframe(scenarios: Sequence[Scenario]) -> pd.DataFrame:
    """Flatten scenarios to the long-format table described in the module docstring."""
    rows = []
    for s in scenarios:
        buses = sorted(set(s.p_load_kw) | set(s.q_load_kvar) | set(s.pv_kw))
        for b in buses:
            rows.append(dict(scenario_id=s.scenario_id, probability=s.probability,
                             period=s.period if s.period is not None else -1,
                             duration_h=s.duration_h if s.duration_h is not None else float("nan"),
                             bus_id=b, p_load_kw=s.p_load_kw.get(b, 0.0),
                             q_load_kvar=s.q_load_kvar.get(b, 0.0), pv_kw=s.pv_kw.get(b, 0.0)))
    return pd.DataFrame(rows)


def scenarios_from_dataframe(df: pd.DataFrame) -> List[Scenario]:
    """Build scenarios from the long-format table (inverse of :func:`scenarios_to_dataframe`)."""
    out: List[Scenario] = []
    for sid, g in df.groupby("scenario_id", sort=False):
        period = None
        if "period" in g and pd.notna(g["period"].iloc[0]) and g["period"].iloc[0] >= 0:
            period = int(g["period"].iloc[0])
        dur = None
        if "duration_h" in g and pd.notna(g["duration_h"].iloc[0]):
            dur = float(g["duration_h"].iloc[0])
        out.append(Scenario(
            scenario_id=str(sid), probability=float(g["probability"].iloc[0]),
            p_load_kw={int(b): float(v) for b, v in zip(g["bus_id"], g["p_load_kw"])},
            q_load_kvar={int(b): float(v) for b, v in zip(g["bus_id"], g["q_load_kvar"])},
            pv_kw={int(b): float(v) for b, v in zip(g["bus_id"], g["pv_kw"]) if v > 0},
            period=period, duration_h=dur))
    return out


def write_scenarios_csv(scenarios: Sequence[Scenario], path: Union[str, Path]) -> None:
    """Write scenarios to CSV."""
    scenarios_to_dataframe(scenarios).to_csv(path, index=False)


def read_scenarios(path: Union[str, Path], sheet: str = "scenarios") -> List[Scenario]:
    """Read scenarios from a CSV (.csv) or Excel (.xlsx/.xls, sheet ``scenarios``) file."""
    p = Path(path)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p, sheet_name=sheet)
    return scenarios_from_dataframe(df)
