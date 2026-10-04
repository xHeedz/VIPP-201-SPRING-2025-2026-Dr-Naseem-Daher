"""
Ablation: no context / 3 environments / retrieved context (model/rag.py).

    python scripts/rag_ablation.py

UAH, leave one driver out: reference = normal windows of the other five drivers; context per window =
road type (trip), speed limit (UAH OSM file, median over the window), manoeuvre (lane change in
EVENTS_LIST_LANE_CHANGES within the window), region spain. AUC normal vs aggressive per held out
driver and pooled.
SUMO planted drivers, split by seed: reference = normal windows of seeds 0 to 4, test on 5 to 9 and the
other way round; context = road type (scenario), traffic state (density level), weather, speed limit,
manoeuvre (lane change in the window). AUC aggressive vs normal pooled over all scenarios (where context
matters) and per scenario.
NGSIM (no labels): share of windows above the 95th percentile of their matched normal windows, with
UAH as the only reference and with UAH + SUMO planted normal drivers as reference.

Outputs data/rag_ablation_uah.csv, data/rag_ablation_sumo.csv, data/rag_ngsim_flags.csv,
docs/hand_checks/rag_explanations.md
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR  # noqa: E402
from datasets.uah import find_trips, load_trip, windows  # noqa: E402
from model.aggressiveness_model import headway_features, index_features, original_score  # noqa: E402
from model.dynamic_weight_agent import DynamicWeightAgent  # noqa: E402
from model.rag import Context, WindowIndex, explain  # noqa: E402
from datasets.contexts import ngsim_context, sumo_context as sumo_ctx, uah_context as uah_ctx  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_uah import auc, train  # noqa: E402

EXT = os.path.abspath(os.path.join(DATA_DIR, "..", "..", "28:9:2026"))
UAH = os.path.join(EXT, "UAH-DRIVESET-v1")
FEATS = ["phi_speed", "phi_accel", "phi_prox", "phi_wave"]


def _window_extras(ps, w, lane_events=None):
    """mean gap (seconds with a leader) and lane change flag for every window row of one vehicle/trip"""
    t = ps["t"].to_numpy()
    gap = ps["gap_m"].to_numpy()
    lane = ps["lane"].to_numpy() if "lane" in ps else None
    g_out, lc = np.full(len(w), np.nan), np.zeros(len(w), dtype=bool)
    for i, t0 in enumerate(w["t0"].to_numpy()):
        a, b = np.searchsorted(t, t0), np.searchsorted(t, t0 + 10.0)
        gg = gap[a:b]
        g_out[i] = gg[gg > 0].mean() if (gg > 0).any() else np.nan
        if lane_events is not None:
            lc[i] = ((lane_events >= t0) & (lane_events < t0 + 10.0)).any()
        elif lane is not None and b > a:
            lc[i] = len(np.unique(lane[a:b])) > 1
    return g_out, lc


# ── UAH ─────────────────────────────────────────────────────────────────────
def uah_windows():
    frames = []
    for trip in find_trips(UAH):
        if trip["behavior"] not in ("normal", "aggressive"):
            continue
        ps = load_trip(trip)
        w = windows(ps, trip)
        if not len(w):
            continue
        w = w[w["speed_kmh"] >= 5.0].reset_index(drop=True)
        wh = windows(ps, trip, features=headway_features)
        wh = wh[wh["t0"].isin(w["t0"])].reset_index(drop=True)
        osm = pd.read_csv(os.path.join(trip["path"], "PROC_OPENSTREETMAP_DATA.txt"), sep=r"\s+", header=None)
        lce = pd.read_csv(os.path.join(trip["path"], "EVENTS_LIST_LANE_CHANGES.txt"), sep=r"\s+", header=None)[0].to_numpy() \
            if os.path.getsize(os.path.join(trip["path"], "EVENTS_LIST_LANE_CHANGES.txt")) else np.array([])
        lim = []
        for t0 in w["t0"]:
            m = (osm[0] >= t0) & (osm[0] < t0 + 10) & (osm[1] > 0)
            lim.append(float(osm.loc[m, 1].median()) if m.any() else (120.0 if trip["road"] == "motorway" else 90.0))
        gap, lc = _window_extras(ps, w, lce)
        frames.append(w.assign(score=original_score(w[FEATS].to_numpy()), score_headway=original_score(wh[FEATS].to_numpy()),
                               limit=lim, gap_m=gap, lane_change=lc))
    return pd.concat(frames, ignore_index=True)


def uah_ablation(w, epochs=600):
    ctx = [uah_ctx(r) for _, r in w.iterrows()]
    y = (w["behavior"] == "aggressive").astype(int).to_numpy()
    rows, pooled = [], {k: np.full(len(w), np.nan) for k in ["none", "env_agent", "rag", "rag_headway", "road_only"]}
    for d in sorted(w["driver"].unique()):
        te, tr = (w["driver"] == d).to_numpy(), (w["driver"] != d).to_numpy()
        ref = [c for c, m in zip(ctx, tr) if m]
        lab = np.where(y[tr] == 1, "aggressive", "normal")
        idx = WindowIndex(w.loc[tr, "score"], ref, lab)
        idx_h = WindowIndex(w.loc[tr, "score_headway"], ref, lab)
        road = WindowIndex(w.loc[tr, "score"], [Context(c.road_type, 100.0) for c in ref], lab)
        cte = [c for c, m in zip(ctx, te) if m]
        agent = train(w[tr], epochs)
        s = {"none": w.loc[te, "score"].to_numpy(),
             "env_agent": np.array([agent.ai(x, "highway" if r == "motorway" else "urban")
                                    for x, r in zip(w.loc[te, FEATS].to_numpy(), w.loc[te, "road"])]),
             "rag": idx.score_many(w.loc[te, "score"].to_numpy(), cte),
             "rag_headway": idx_h.score_many(w.loc[te, "score_headway"].to_numpy(), cte),
             "road_only": road.score_many(w.loc[te, "score"].to_numpy(), [Context(c.road_type, 100.0) for c in cte])}
        for k, v in s.items():
            pooled[k][te] = v
        rows.append({"held_out_driver": d, **{f"auc_{k}": auc(v, y[te]) for k, v in s.items()}})
    r = pd.DataFrame(rows)
    r.loc[len(r)] = {"held_out_driver": "pooled", **{f"auc_{k}": auc(v, y) for k, v in pooled.items()}}
    return r, pooled


# ── SUMO ────────────────────────────────────────────────────────────────────
def sumo_windows():
    from datasets.sumo_log import windows_from_per_second
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "sumo_planted", "*.per_second.csv.gz"))):
        sc, dens, seed = os.path.basename(f).replace(".per_second.csv.gz", "").rsplit("_", 2)
        ps = pd.read_csv(f)
        w = windows_from_per_second(ps, sc, "x")
        wh = windows_from_per_second(ps, sc, "x", features=headway_features)
        lc = np.zeros(len(w), dtype=bool)
        gap = np.full(len(w), np.nan)
        pos = w.groupby("trip").indices
        for vid, g in ps.groupby("vehicle_id", sort=False):
            if vid in pos:
                ix = pos[vid]
                gap[ix], lc[ix] = _window_extras(g, w.iloc[ix])
        frames.append(w.assign(scenario=sc, density=dens, seed=int(seed[1:]), score=original_score(w[FEATS].to_numpy()),
                               score_headway=original_score(wh[FEATS].to_numpy()), gap_m=gap, lane_change=lc))
        print("  ", os.path.basename(f), flush=True)
    return pd.concat(frames, ignore_index=True)


def sumo_ablation(w):
    w = w[w["behavior"].isin(["normal", "aggressive"])].reset_index(drop=True)
    ctx = [sumo_ctx(r) for _, r in w.iterrows()]
    y = (w["behavior"] == "aggressive").astype(int).to_numpy()
    agent = DynamicWeightAgent.load(os.path.join(DATA_DIR, "dynamic_weight_agent.json"))
    env = w["scenario"].map({"highway": "highway", "jam": "highway", "merge": "highway", "weather": "weather",
                             "urban": "urban", "roundabout": "urban"}).to_numpy()
    out = {k: np.full(len(w), np.nan) for k in ["none", "none_headway", "env_agent", "rag", "rag_headway"]}
    out["none"], out["none_headway"] = w["score"].to_numpy(), w["score_headway"].to_numpy()
    out["env_agent"] = np.array([agent.ai(x, e) for x, e in zip(w[FEATS].to_numpy(), env)])
    for half in (0, 1):
        ref_m = ((w["seed"] < 5) == (half == 0)).to_numpy()
        te = ~ref_m
        ref = [c for c, m in zip(ctx, ref_m) if m]
        lab = np.where(y[ref_m] == 1, "aggressive", "normal")
        cte = [c for c, m in zip(ctx, te) if m]
        out["rag"][te] = WindowIndex(w.loc[ref_m, "score"], ref, lab).score_many(w.loc[te, "score"].to_numpy(), cte)
        out["rag_headway"][te] = WindowIndex(w.loc[ref_m, "score_headway"], ref, lab).score_many(
            w.loc[te, "score_headway"].to_numpy(), cte)
    rows = [{"scenario": "pooled", **{f"auc_{k}": auc(v, y) for k, v in out.items()}}]
    for (sc, d), g in w.groupby(["scenario", "density"]):
        ix = g.index.to_numpy()
        rows.append({"scenario": f"{sc} {d}", **{f"auc_{k}": auc(v[ix], y[ix]) for k, v in out.items()}})
    return pd.DataFrame(rows), w, out, ctx


# ── NGSIM flags ─────────────────────────────────────────────────────────────
def ngsim_flags(uah_w, sumo_w):
    path = os.path.join(DATA_DIR, "unlabelled_windows.csv.gz")
    if not os.path.exists(path):
        return None
    u = pd.read_csv(path)
    ng = u[u["site"].str.startswith("ngsim")].reset_index(drop=True)

    ctx = [ngsim_context(r) for _, r in ng.iterrows()]
    un = uah_w[uah_w["behavior"] == "normal"]
    sn = sumo_w[sumo_w["behavior"] == "normal"]
    refs = {"uah only": (un["score"], [uah_ctx(r) for _, r in un.iterrows()]),
            "uah + sumo": (pd.concat([un["score"], sn["score"]]),
                           [uah_ctx(r) for _, r in un.iterrows()] + [sumo_ctx(r) for _, r in sn.iterrows()])}
    rows = []
    for name, (s, c) in refs.items():
        idx = WindowIndex(s, c)
        pct = idx.score_many(ng["score"].to_numpy(), ctx)
        for site, g in ng.assign(pct=pct).groupby("site"):
            rows.append({"reference": name, "site": site, "windows": len(g), "share_flagged_95": float((g["pct"] >= 95).mean()),
                         "share_raw_ge_70": float((g["score"] >= 70).mean()), "median_pct": float(g["pct"].median())})
    return pd.DataFrame(rows)


def explanations(uah_w, sumo_w):
    """Three worked explanations, each scored against a reference that excludes its own driver / seed half."""
    L = ["# context score explanations (examples)", "",
         "generated by `scripts/rag_ablation.py` with `model/rag.py` `explain`; rules from `knowledge/`.", ""]

    def one(title, r, ref_w, ctx_fn, head):
        ref = ref_w[ref_w["behavior"] == "normal"]
        idx = WindowIndex(ref["score"], [ctx_fn(x) for _, x in ref.iterrows()])
        c = ctx_fn(r)
        pct, thr, lvl, n = idx.score(r["score"], c)
        return [f"## {title}", "", head, f"raw index {r['score']:.1f}", "",
                explain(r[FEATS].to_numpy(float), c, pct, thr, r["speed_kmh"], r["gap_m"], lvl, n), ""]

    agg = uah_w[uah_w["behavior"] == "aggressive"].sort_values("score")
    r = agg.iloc[int(len(agg) * 0.95)]
    L += one("UAH aggressive window, 95th percentile of raw scores", r, uah_w[uah_w["driver"] != r["driver"]], uah_ctx,
             f"trip `{r['trip']}`, t0 {r['t0']:.1f} s, reference: normal windows of the other five drivers. ")
    nor = uah_w[uah_w["behavior"] == "normal"].sort_values("score")
    r = nor.iloc[len(nor) // 2]
    L += one("UAH normal window, median raw score", r, uah_w[uah_w["driver"] != r["driver"]], uah_ctx,
             f"trip `{r['trip']}`, t0 {r['t0']:.1f} s, reference: normal windows of the other five drivers. ")
    jam = sumo_w[(sumo_w["scenario"] == "jam") & (sumo_w["behavior"] == "normal") & (sumo_w["seed"] >= 5)].sort_values("score")
    r = jam.iloc[int(len(jam) * 0.9)]
    L += one("SUMO jam, normal driver, 90th percentile of raw scores", r, sumo_w[sumo_w["seed"] < 5], sumo_ctx,
             f"vehicle `{r['trip']}` (seed {r['seed']}), reference: normal windows of seeds 0 to 4. ")
    with open(os.path.join(DATA_DIR, "..", "docs", "hand_checks", "rag_explanations.md"), "w") as f:
        f.write("\n".join(L))


def ngsim_flags_headway(uah_w, sumo_w):
    """as ngsim_flags, with the time headway variant as the raw score for the windows and the references"""
    u = pd.read_csv(os.path.join(DATA_DIR, "unlabelled_windows.csv.gz"))
    ng = u[u["site"].str.startswith("ngsim")].reset_index(drop=True)
    ctx = [ngsim_context(r) for _, r in ng.iterrows()]
    un, sn = uah_w[uah_w["behavior"] == "normal"], sumo_w[sumo_w["behavior"] == "normal"]
    rows = []
    for name, s, c in [("uah only", un["score_headway"], [uah_ctx(r) for _, r in un.iterrows()]),
                       ("uah + sumo", pd.concat([un["score_headway"], sn["score_headway"]]),
                        [uah_ctx(r) for _, r in un.iterrows()] + [sumo_ctx(r) for _, r in sn.iterrows()])]:
        pct = WindowIndex(s, c).score_many(ng["score_headway"].to_numpy(), ctx)
        for site, g in ng.assign(pct=pct).groupby("site"):
            rows.append({"reference": name, "site": site, "share_flagged_95": float((g["pct"] >= 95).mean()),
                         "share_raw_ge_70": float((g["score_headway"] >= 70).mean()), "median_pct": float(g["pct"].median())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    uw = uah_windows()
    ru, pooled = uah_ablation(uw)
    ru.to_csv(os.path.join(DATA_DIR, "rag_ablation_uah.csv"), index=False)
    print(ru.round(3).to_string(index=False), flush=True)
    sw = sumo_windows()
    rs, sw2, out, ctx = sumo_ablation(sw)
    rs.to_csv(os.path.join(DATA_DIR, "rag_ablation_sumo.csv"), index=False)
    print(rs.round(3).to_string(index=False), flush=True)
    nf = ngsim_flags(uw, sw2)
    if nf is not None:
        nf.to_csv(os.path.join(DATA_DIR, "rag_ngsim_flags.csv"), index=False)
        print(nf.round(3).to_string(index=False))
    explanations(uw, sw2)
    nh = ngsim_flags_headway(uw, sw2)
    nh.to_csv(os.path.join(DATA_DIR, "rag_ngsim_flags_headway.csv"), index=False)
    print(nh.round(3).to_string(index=False))
