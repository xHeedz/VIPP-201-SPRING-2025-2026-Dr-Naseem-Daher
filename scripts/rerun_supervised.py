"""
The supervised model (DynamicWeightAgent) on the cleaned data and the reference terms (speed, accel^2, prox^2 mix,
wave; unknown terms left out of the score with the remaining weights rescaled, as in the reference formula).

    python scripts/rerun_supervised.py [--epochs 600]

UAH leave one driver out (train on 5 drivers' normal and aggressive windows, test on the 6th), against the
reference formula on the same windows (agent AUC on the margin: score minus
the threshold of the window's environment; the raw pooled AUC is kept for comparison); then the agent trained on all UAH drivers applied to the planted test
seeds 7 to 9 (aggressive vs normal). Outputs data/rerun_supervised.csv.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402
from datasets.windows import TERM_COLS  # noqa: E402
from model.aggressiveness_model import REF_CUTOFFS, reference_score  # noqa: E402
from model.dynamic_weight_agent import ENVIRONMENTS, DynamicWeightAgent  # noqa: E402
from rerun_formula_all import auc  # noqa: E402

CLEAN = os.path.join(DATA_DIR, "clean")


class KnownTermsAgent(DynamicWeightAgent):
    """score = 100 x weighted mean of the known terms (NaN terms left out); training input NaN = unknown."""

    def score(self, env_type, metrics):
        m = torch.nan_to_num(metrics, nan=0.0)
        k = (~torch.isnan(metrics)).float()
        w = self.get_weights(env_type)
        return 100.0 * (w * m * k).sum(-1) / (w * k).sum(-1).clamp_min(1e-6)

    def ai_known(self, X, env):
        with torch.no_grad():
            return self.score(env, torch.tensor(X, dtype=torch.float32)).numpy()


def agent_scores(agent, w, margin=True):
    """score minus the learned threshold of the window's environment (margin): the agent learns one threshold per
    environment, so raw scores of different environments are on different scales and must not be ranked together"""
    s = np.full(len(w), np.nan)
    for e in w["environment"].unique():
        m = (w["environment"] == e).to_numpy()
        s[m] = agent.ai_known(w.loc[m, TERM_COLS].to_numpy(), e) - (agent.threshold(e) if margin else 0.0)
    return s


def train(w, epochs):
    a = KnownTermsAgent()
    a.fit_labels(w[TERM_COLS].to_numpy(), (w["label"] == "aggressive").astype(int), w["environment"], epochs=epochs)
    return a


def accuracy(scores, w, thr):
    """window accuracy aggressive vs normal; thr: scalar or {environment: threshold}"""
    t = w["environment"].map(thr).to_numpy() if isinstance(thr, dict) else thr
    return float(((scores >= t) == (w["label"] == "aggressive")).mean())


def main(epochs):
    uah = pd.read_csv(os.path.join(CLEAN, "windows_uah.csv.gz"))
    uah = uah[uah["label"].isin(["normal", "aggressive"])].reset_index(drop=True)
    uah["formula"] = reference_score(uah[TERM_COLS].to_numpy())
    rows = []
    for drv in sorted(uah["group"].unique()):
        tr, te = uah[uah["group"] != drv], uah[uah["group"] == drv]
        a = train(tr, epochs)
        s = agent_scores(a, te)
        raw = agent_scores(a, te, margin=False)
        y = te["label"] == "aggressive"
        rows.append({"test": "uah lodo", "held_out": drv, "windows": len(te),
                     "supervised_auc": auc(s[y.to_numpy()], s[~y.to_numpy()]),
                     "supervised_auc_raw_pooled": auc(raw[y.to_numpy()], raw[~y.to_numpy()]),
                     "supervised_acc": accuracy(s, te, 0.0),
                     "formula_auc": auc(te.loc[y, "formula"], te.loc[~y, "formula"]),
                     "formula_acc": accuracy(te["formula"].to_numpy(), te, REF_CUTOFFS[1]),
                     "always_normal_acc": float((~y).mean()),
                     "weights": "; ".join(f"{e} " + "/".join(f"{x:.2f}" for x in a.weights_np(e)) for e in ("highway", "urban"))})
        print(rows[-1], flush=True)
    final = train(uah, epochs)
    pl = pd.read_csv(os.path.join(CLEAN, "windows_planted.csv.gz"))
    pl = pl[pl["group"].isin([7, 8, 9]) & pl["label"].isin(["normal", "aggressive"])]
    s = agent_scores(final, pl)
    y = (pl["label"] == "aggressive").to_numpy()
    f = reference_score(pl[TERM_COLS].to_numpy())
    rows.append({"test": "planted test seeds, agent trained on all UAH", "held_out": "seeds 7-9", "windows": len(pl),
                 "supervised_auc": auc(s[y], s[~y]), "formula_auc": auc(f[y], f[~y])})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, "rerun_supervised.csv"), index=False)
    print(res.drop(columns=["weights"], errors="ignore").groupby("test").mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=600)
    main(p.parse_args().epochs)
