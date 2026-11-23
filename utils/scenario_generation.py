"""Scenario generation for the stochastic model and forecast points for the robust model.

All scenarios are returned as :class:`data.load_scenarios.Scenario` (kW / kVAr).
Reactive load always follows active load with the bus's forecast power factor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from data.load_scenarios import Scenario, load_forecast_kw, normalize_probabilities
from data.pv_scenarios import pv_forecast_kw, pv_with_deviation


def forecast_points(net, periods: Sequence[int] = (12, 19),
                    weights: Optional[Sequence[float]] = None) -> List[Scenario]:
    """Nominal (forecast) operating points, one per hour in ``periods``.

    Their ``probability`` is the period weight (equal by default) and is used by the robust
    model to weight the worst-case cost of each period.
    """
    w = list(weights) if weights is not None else [1.0] * len(periods)
    if len(w) != len(periods):
        raise ValueError("weights and periods must have the same length")
    out = []
    for h, wi in zip(periods, w):
        p, q = load_forecast_kw(net, h)
        pv = pv_forecast_kw(net, h)
        out.append(Scenario(scenario_id=f"h{h:02d}_forecast", probability=float(wi),
                            p_load_kw=p, q_load_kvar=q, pv_kw=pv, period=h))
    return normalize_probabilities(out)


def _deviate(net, base: Scenario, sid: str, prob: float, load_factor: float, pv_factor: float,
             load_dev: float, pv_dev: float) -> Scenario:
    """Scale loads by (1 + load_dev*load_factor) and PV by (1 + pv_dev*pv_factor), PV clipped to capacity."""
    p = {b: v * (1.0 + load_dev * load_factor) for b, v in base.p_load_kw.items()}
    q = {b: v * (1.0 + load_dev * load_factor) for b, v in base.q_load_kvar.items()}
    pv = {b: pv_with_deviation(v, net.pu_to_kw(net.pv_cap_pu[b]), pv_dev, pv_factor)
          for b, v in base.pv_kw.items()}
    return Scenario(scenario_id=sid, probability=prob, p_load_kw=p, q_load_kvar=q, pv_kw=pv,
                    period=base.period)


def generate_structured_scenarios(net, periods: Sequence[int] = (12, 19), load_dev: float = 0.15,
                                  pv_dev: float = 0.30,
                                  period_weights: Optional[Sequence[float]] = None,
                                  case_probs: Sequence[float] = (0.25, 0.50, 0.25)) -> List[Scenario]:
    """Three-case scenario tree per period: adverse / nominal / benign.

    * adverse : load +load_dev, PV -pv_dev   (stresses voltage/ampacity)
    * nominal : forecast
    * benign  : load -load_dev, PV +pv_dev   (stresses over-voltage / curtailment)

    Scenario probability = period weight x case probability (sums to one).
    """
    cases = (("adverse", +1.0, -1.0), ("nominal", 0.0, 0.0), ("benign", -1.0, +1.0))
    out: List[Scenario] = []
    for base in forecast_points(net, periods, period_weights):
        for (name, lf, pf), cp in zip(cases, case_probs):
            out.append(_deviate(net, base, f"h{base.period:02d}_{name}", base.probability * cp,
                                lf, pf, load_dev, pv_dev))
    return normalize_probabilities(out)


def generate_random_scenarios(net, n_per_period: int = 5, periods: Sequence[int] = (12, 19),
                              load_dev: float = 0.15, pv_dev: float = 0.30, correlation: float = 0.7,
                              seed: int = 2024, probabilities: Optional[Sequence[float]] = None,
                              period_weights: Optional[Sequence[float]] = None) -> List[Scenario]:
    """Random scenarios with a system-wide common factor and bus-level noise.

    For each scenario: load_factor_i = rho*c_L + sqrt(1-rho^2)*e_i with c_L, e_i ~ U[-1, 1]
    (clipped to [-1, 1]); the same construction is used for PV.  Deviation magnitudes are
    ``load_dev`` and ``pv_dev`` (fraction of forecast).  ``probabilities`` (length =
    total scenarios) overrides the default equal split (period weight / n_per_period).
    The generator is seeded, hence reproducible.
    """
    rng = np.random.default_rng(seed)
    rho = float(correlation)
    out: List[Scenario] = []
    for base in forecast_points(net, periods, period_weights):
        for k in range(n_per_period):
            cl = rng.uniform(-1, 1)
            cp = rng.uniform(-1, 1)
            p, q, pv = {}, {}, {}
            for b, v in base.p_load_kw.items():
                f = float(np.clip(rho * cl + np.sqrt(1 - rho ** 2) * rng.uniform(-1, 1), -1, 1))
                p[b] = v * (1 + load_dev * f)
                q[b] = base.q_load_kvar[b] * (1 + load_dev * f)
            for b, v in base.pv_kw.items():
                f = float(np.clip(rho * cp + np.sqrt(1 - rho ** 2) * rng.uniform(-1, 1), -1, 1))
                pv[b] = pv_with_deviation(v, net.pu_to_kw(net.pv_cap_pu[b]), pv_dev, f)
            out.append(Scenario(scenario_id=f"h{base.period:02d}_rnd{k + 1}",
                                probability=base.probability / n_per_period,
                                p_load_kw=p, q_load_kvar=q, pv_kw=pv, period=base.period))
    if probabilities is not None:
        if len(probabilities) != len(out):
            raise ValueError("probabilities must have one entry per generated scenario")
        for s, pr in zip(out, probabilities):
            s.probability = float(pr)
    return normalize_probabilities(out)
