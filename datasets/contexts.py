"""
Context (model/rag.py) of a window from each data source: the retrieval key for the context score.

    uah_context      road type from the trip, speed limit from PROC_OPENSTREETMAP_DATA (median over the
                     window), lane change from EVENTS_LIST_LANE_CHANGES, region spain, traffic unknown
    sumo_context     planted SUMO runs: road type and limit of the scenario, traffic state from the demand
                     level, rain for the weather scenario, lane change from the lane id, region simulation
    ngsim_context    road type (freeway -> motorway, arterial -> intersection), typical US limits (not in the
                     data), traffic state from density (veh/km/lane: < 20 free, < 40 dense, else jam)
    pneuma_context   downtown Athens, 50 km/h (2018), traffic state from vehicles within 50 m
Each takes one window row (with the columns the window tables of scripts/rag_ablation.py and
scripts/score_unlabelled.py carry).
"""
import numpy as np

from model.rag import Context

SUMO_ROAD = {"highway": ("motorway", 120.0), "jam": ("motorway", 120.0), "weather": ("motorway", 120.0),
             "merge": ("merge", 100.0), "urban": ("intersection", 50.0), "roundabout": ("roundabout", 50.0)}
SUMO_STATE = {"low": "free", "medium": "dense", "jam": "jam"}


def _state_from_density(d, free=20.0, jam=40.0):
    if not np.isfinite(d):
        return "unknown"
    return "free" if d < free else ("dense" if d < jam else "jam")


def uah_context(r):
    return Context(road_type=r["road"], speed_limit_kmh=float(r["limit"]),
                   manoeuvre="lane_change" if r["lane_change"] else "cruise", region="spain")


def sumo_context(r):
    road, lim = SUMO_ROAD[r["scenario"]]
    return Context(road_type=road, speed_limit_kmh=lim, traffic_state=SUMO_STATE.get(r["density"], "dense"),
                   manoeuvre="lane_change" if r["lane_change"] else "cruise",
                   weather="rain" if r["scenario"] == "weather" else "clear", region="simulation")


def ngsim_context(r):
    freeway = r["road_type"] == "freeway"
    return Context(road_type="motorway" if freeway else "intersection", speed_limit_kmh=105.0 if freeway else 56.0,
                   traffic_state=_state_from_density(r["density"]), region="united_states",
                   density=float(r["density"]) if np.isfinite(r["density"]) else float("nan"))


def pneuma_context(r, q=(9.0, 16.0)):
    d = r["density"]
    st = "unknown" if not np.isfinite(d) else ("free" if d <= q[0] else ("dense" if d <= q[1] else "jam"))
    return Context(road_type="downtown", speed_limit_kmh=50.0, traffic_state=st, region="greece")
