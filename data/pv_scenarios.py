"""PV profiles and PV forecast / deviation helpers. Power in kW."""
from __future__ import annotations

import math
from typing import Dict, List

# Clear-sky style normalised PV profile (fraction of installed capacity), hours 0..23.
# Half-sine between 06:00 and 18:00, derated by 0.95 (soiling/inverter losses; assumed).
PV_PROFILES: Dict[str, List[float]] = {
    "pv_clear_sky": [0.95 * max(0.0, math.sin(math.pi * (h + 0.5 - 6.0) / 12.0))
                     if 6 <= h + 0.5 <= 18 else 0.0 for h in range(24)],
    "pv_flat": [1.0] * 24,
}


def pv_forecast_kw(net, hour: int) -> Dict[int, float]:
    """Forecast available PV power [kW] per PV bus for a given hour (0-23)."""
    out: Dict[int, float] = {}
    for _, row in net.buses.iterrows():
        cap = float(row["pv_capacity_kw"])
        if cap <= 0:
            continue
        prof = PV_PROFILES.get(str(row["pv_profile"]), PV_PROFILES["pv_clear_sky"])
        out[int(row["bus_id"])] = cap * prof[hour % len(prof)]
    return out


def pv_with_deviation(forecast_kw: float, capacity_kw: float, deviation: float, factor: float) -> float:
    """Apply a relative PV deviation to a forecast.

    ``deviation`` is the maximum relative deviation (e.g. 0.3 = +-30 %), ``factor`` in [-1, 1]
    selects the realisation (+1 = full upward deviation).  The result is clipped to
    [0, capacity_kw].
    """
    return min(max(forecast_kw * (1.0 + deviation * factor), 0.0), capacity_kw)
