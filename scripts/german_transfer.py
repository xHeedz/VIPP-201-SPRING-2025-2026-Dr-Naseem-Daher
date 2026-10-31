"""
Moving the labeling agent to German roads (Dr. Daher, Oct 2026): what does it need to work there?

    python scripts/german_transfer.py [--steps 1000000] [--skip-agents]

Simulated part (true labels): the planted drivers of the other scenarios (data/clean/planted.csv.gz) and of the German
style autobahn scenario (data/clean/planted_de.csv.gz, trucks not scored). PPO, risk weighted reward, wait 1 s / 5 s;
test always on autobahn seeds 7 to 9, never seen in training:
  A_naive        trained on the other roads only, the 8 inputs of the earlier best agent
  B_relative     trained on the other roads only, plus speed vs limit / vs traffic, IDM residual, danger
  C_de_onehot    trained on the other roads and autobahn seeds 0 to 6, B's inputs plus the country one hot (country_de)
  D_de_only      trained on autobahn seeds 0 to 6 only, B's inputs
Also the reference and extended formula on the autobahn test windows (AUC aggressive vs normal).
Real highD part (no labels): proxy labels from rule violations per 10 s window: aggressive when at least half the seconds
are below 0.9 s headway or at least 20% over the speed limit; calm when neither happens while following a car.
Reference and extended formula scored against them, with all terms and without the gap and speed vs limit terms
(those overlap with how the proxy labels are defined).
Outputs data/german_transfer.csv, data/german_transfer_highd_proxy.csv.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from datasets.signals import EXT_COLS, add_signals  # noqa: E402
from datasets.windows import TERM_COLS, per_second_terms, windows  # noqa: E402
from env.labeling_ref import COUNTRY_FEATURES, EXT_FEATURES, LabelingRefEnv, evaluate, run_policy, sequences_from_clean  # noqa: E402
from fit_reference_weights import auc_cols, scores_all  # noqa: E402
from model.aggressiveness_model import REF_WEIGHTS  # noqa: E402
from state_reward_ablation import CLASSES, STATES, TEST_SEEDS, train_agent  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")
BASE = STATES["small6"]
REL = BASE + EXT_FEATURES


def auc(s, y):
    return float(auc_cols(np.asarray(s, float)[:, None], np.asarray(y))[0])


def formula_weights():
    ref = np.r_[np.asarray(REF_WEIGHTS), np.zeros(len(EXT_COLS))]
    ext = np.asarray(json.load(open(os.path.join(DATA_DIR, "extended_weights.json")))["weights"])
    return ref, ext


def agents(steps):
    other = sequences_from_clean(pd.read_csv(os.path.join(CLEAN, "planted.csv.gz")), extended=True)
    de = [s for s in sequences_from_clean(pd.read_csv(os.path.join(CLEAN, "planted_de.csv.gz")), extended=True)
          if s["label"] in CLASSES]
    o_tr = [s for s in other if s["group"] not in TEST_SEEDS]
    de_tr = [s for s in de if s["group"] not in TEST_SEEDS]
    de_te = [s for s in de if s["group"] in TEST_SEEDS and len(s["terms"]) >= 60]
    o_te = [s for s in other if s["group"] in TEST_SEEDS and len(s["terms"]) >= 60]
    print(f"other train {len(o_tr)}, autobahn train {len(de_tr)}, autobahn test {len(de_te)}", flush=True)
    rows = []
    runs = [("A_naive", o_tr, BASE), ("B_relative", o_tr, REL), ("C_de_onehot", o_tr + de_tr, REL + COUNTRY_FEATURES),
            ("D_de_only", de_tr, REL)]
    for name, tr, feats in runs:
        t = time.time()
        cfg = dict(features=feats, wait5=True, risk=True)
        model = train_agent("ppo", LabelingRefEnv(tr, CLASSES, **cfg), steps, 0)
        model.save(os.path.join(DATA_DIR, f"german_{name}.zip"))
        for test_name, te in (("autobahn test", de_te), ("other roads test", o_te)):
            if test_name == "other roads test" and name == "D_de_only":
                continue
            _, m = evaluate(te, *run_policy(model, LabelingRefEnv(te, CLASSES, **cfg), len(te)))
            rows.append({"agent": name, "test": test_name, "inputs": len(feats), **m})
            print(f"{name} on {test_name}: balanced {m['balanced_acc']:.3f}, {m['seconds']:.1f} s ({time.time() - t:.0f} s)",
                  flush=True)
    return rows


def formulas_on_autobahn():
    w = windows(pd.read_csv(os.path.join(CLEAN, "planted_de.csv.gz")), extended=True)
    w = w[w["group"].isin(TEST_SEEDS) & w["label"].isin(["aggressive", "normal"])]
    ref, ext = formula_weights()
    T = w[TERM_COLS + EXT_COLS].to_numpy()
    y = (w["label"] == "aggressive").to_numpy(int)
    return [{"agent": "formula reference", "test": "autobahn test windows", "auc": auc(scores_all(T, ref[None])[:, 0], y)},
            {"agent": "formula extended", "test": "autobahn test windows", "auc": auc(scores_all(T, ext[None])[:, 0], y)}]


def highd_proxy():
    d = pd.read_csv(os.path.join(CLEAN, "highd.csv.gz"))
    d = add_signals(per_second_terms(d))
    v = d["speed_kmh"] / 3.6
    thw = np.where(d["gap_m"] > 0, d["gap_m"] / v.clip(lower=0.5), np.inf)
    d["tailgate"] = (thw < 0.9).astype(float)
    d["speeding"] = (d["rel_limit"] >= 0.2).astype(float)
    d["following"] = (thw < 3.0).astype(float)
    d["win"] = ((d["t"] - d.groupby("vid")["t"].transform("min")) // 10).astype(int)
    g = d.groupby(["vid", "win"])
    w = g[TERM_COLS + EXT_COLS + ["tailgate", "speeding", "following"]].mean()
    w = w[g.size() >= 8]
    aggr = (w["tailgate"] >= 0.5) | (w["speeding"] >= 0.5)
    calm = (w["tailgate"] == 0) & (w["speeding"] == 0) & (w["following"] >= 0.5)
    w = w[aggr | calm]
    y = aggr[w.index].to_numpy(int)
    ref, ext = formula_weights()
    T = w[TERM_COLS + EXT_COLS].to_numpy()
    rows = [{"proxy": "tailgating or speeding", "windows": len(w), "share_proxy_aggressive": float(y.mean())}]
    for name, wt in (("reference", ref), ("extended", ext)):
        rows[0][f"auc_{name}_all_terms"] = auc(scores_all(T, wt[None])[:, 0], y)
        wo = wt.copy()
        wo[[2, 5]] = 0.0                        # without gap and speed vs limit (they define the proxy)
        rows[0][f"auc_{name}_without_gap_and_limit"] = auc(scores_all(T, wo[None])[:, 0], y)
    for i, n in enumerate(["speed", "accel2", "gap2", "lane", "ttc", "rel_limit", "rel_traffic", "idm"]):
        rows[0][f"single_{n}"] = auc(T[:, i], y)
    return rows


def main(steps, skip_agents):
    rows = [] if skip_agents else agents(steps)
    rows += formulas_on_autobahn()
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "german_transfer.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    hp = pd.DataFrame(highd_proxy())
    hp.to_csv(os.path.join(DATA_DIR, "german_transfer_highd_proxy.csv"), index=False)
    print(hp.round(3).T.to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--skip-agents", action="store_true")
    a = p.parse_args()
    main(a.steps, a.skip_agents)
