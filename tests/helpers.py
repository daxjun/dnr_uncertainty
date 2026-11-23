"""Small test feeders."""
import pandas as pd

from data.network_data import Network
from data.load_scenarios import Scenario


def small_network(loads=None, pv=None, v_root=1.0) -> Network:
    """6-bus meshed feeder: tree 1-2-3-4, 1-5-6 plus ties 4-6 (open) and 3-5 (open)."""
    loads = loads or {2: 300, 3: 400, 4: 500, 5: 300, 6: 400}
    pv = pv or {}
    rows = []
    for b in range(1, 7):
        p = loads.get(b, 0.0)
        rows.append(dict(bus_id=b, bus_type="root" if b == 1 else "load", p_load_kw=p, q_load_kvar=0.4 * p,
                         v_min_pu=0.95, v_max_pu=1.05, pv_capacity_kw=pv.get(b, 0.0),
                         load_profile="flat", pv_profile="pv_flat" if b in pv else ""))
    edges = [(1, 1, 2, 1.0, 0.8, 1), (2, 2, 3, 1.2, 0.9, 1), (3, 3, 4, 1.5, 1.0, 1), (4, 1, 5, 1.3, 0.9, 1),
             (5, 5, 6, 1.6, 1.0, 1), (6, 4, 6, 0.8, 0.6, 0), (7, 3, 5, 0.7, 0.5, 0)]
    br = [dict(branch_id=i, from_bus=f, to_bus=t, r_ohm=r, x_ohm=x, i_max_a=300.0, initial_status=s,
               normally_open=1 - s, switchable=1, switch_cost=1.0) for i, f, t, r, x, s in edges]
    return Network(pd.DataFrame(rows), pd.DataFrame(br), s_base_mva=10.0, v_base_kv=12.66, root_bus=1,
                   v_root_pu=v_root, name="test6")


def scen_from_net(net, sid="s", prob=1.0, load_f=1.0, pv_f=1.0, pv_kw=None):
    p = {b: net.pu_to_kw(net.p_load_pu[b]) * load_f for b in net.bus_ids}
    q = {b: net.pu_to_kw(net.q_load_pu[b]) * load_f for b in net.bus_ids}
    pv = pv_kw if pv_kw is not None else {b: net.pu_to_kw(net.pv_cap_pu[b]) * pv_f for b in net.pv_buses}
    return Scenario(sid, prob, p, q, pv)
