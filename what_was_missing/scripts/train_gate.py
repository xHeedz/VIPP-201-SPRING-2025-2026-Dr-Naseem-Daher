"""
Train the DynamicWeightAgent and export the gate head as model/gate_v2.pt.

RECREATED BY CLAUDE (Anthropic), September 2026. See ../README.md.

Training follows the final report: KL to the per-regime target weights plus the
RL term on non-crash samples, alpha annealed from 1.0 to 0.5 over 50 epochs,
200 epochs in total. Each telemetry row is labeled with the regime of the
scenario it came from (highway -> highway, intersection -> urban,
low_friction -> weather).

    python scripts/record_telemetry.py --scenario highway          # and the other two
    python scripts/train_gate.py results/telemetry_*.csv
"""
import argparse
import csv
import datetime
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from agent.aggressiveness import context_vector, normalize_features  # noqa: E402
from agent.dynamic_weight_agent import (  # noqa: E402
    REGIME_TARGETS, REGIMES, DynamicWeightAgent, agent_loss, alpha_schedule, export_gate,
)

SCENARIO_REGIME = {"highway": "highway", "intersection": "urban", "low_friction": "weather"}


def load(paths):
    feats, ctxs, regimes, crashed = [], [], [], []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                feats.append(normalize_features(float(r["speed"]), float(r["accel"]), float(r["prox"]), float(r["wave"])))
                ctxs.append(context_vector(r["road_type"], float(r["density"]), float(r["friction"]), float(r["slip"])))
                regimes.append(REGIMES.index(SCENARIO_REGIME[r["scenario"]]))
                crashed.append(r["crashed"] == "1")
    return (torch.tensor(feats), torch.tensor(ctxs), torch.tensor(regimes), torch.tensor(crashed))


def train(paths, head="gate", epochs=200, lr=1e-2, batch=512, seed=0, alpha_floor=0.5):
    torch.manual_seed(seed)
    X, C, R, crashed = load(paths)
    targets = REGIME_TARGETS[R]
    agent = DynamicWeightAgent(head)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    history = []
    n = len(X)
    for epoch in range(epochs):
        alpha = alpha_schedule(epoch, floor=alpha_floor)
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            w, _ = agent(X[idx], C[idx])
            loss, kl, rl = agent_loss(w, targets[idx], X[idx], alpha, crashed[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        history.append((epoch, alpha, tot / n))
    return agent, (X, C, R), history


def summarize(agent, data):
    X, C, R = data
    agent.eval()
    out = {}
    with torch.no_grad():
        w, mix = agent(X, C)
        for k, name in enumerate(REGIMES):
            m = R == k
            out[name] = {"mean_weights": [round(v, 3) for v in w[m].mean(0).tolist()],
                         "target": [round(v, 3) for v in REGIME_TARGETS[k].tolist()],
                         "l1_to_target": round((w[m].mean(0) - REGIME_TARGETS[k]).abs().sum().item(), 4),
                         "mean_mixture": [round(v, 3) for v in mix[m].mean(0).tolist()]}
    return out


def handoff(agent, steps=40):
    """Mixture along a context path from pure highway to pure weather."""
    a = torch.tensor([1.0, 0.0, 0.25, 1.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.25, 0.4, 0.3])
    path = torch.stack([a + (b - a) * t / (steps - 1) for t in range(steps)])
    with torch.no_grad():
        _, mix = agent(torch.zeros(steps, 4), path)
    return mix


def plot(history, mix, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h[0] for h in history], [h[2] for h in history], color="#840032")
    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch")
    for k, name in enumerate(REGIMES):
        ax[1].plot(mix[:, k].numpy(), label=name)
    ax[1].set_title("gate mixture, context moving from dry highway to low friction")
    ax[1].set_xlabel("step along the context path")
    ax[1].legend()
    fig.suptitle("recreated by Claude, trained on SUMO telemetry recorded in September 2026", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--head", default="gate", choices=["gate", "mlp"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--alpha-floor", type=float, default=0.5,
                    help="0.5 follows the report; 1.0 trains on the KL anchor alone")
    ap.add_argument("--out", default=os.path.join(ROOT, "model", "gate_v2.pt"))
    a = ap.parse_args()
    agent, data, history = train(a.csv, a.head, a.epochs, alpha_floor=a.alpha_floor)
    summary = summarize(agent, data)
    print(f"final loss {history[-1][2]:.4f} (alpha {history[-1][1]:.2f})")
    for name, s in summary.items():
        print(f"  {name:8s} weights {s['mean_weights']}  target {s['target']}  L1 {s['l1_to_target']}  mixture {s['mean_mixture']}")
    if a.head == "gate":
        provenance = {
            "created_by": "Claude (Anthropic), recreation of a file described in the VIPP 201A reports",
            "created_on": datetime.date.today().isoformat(),
            "not_the_original": True,
            "training_data": [os.path.basename(p) for p in a.csv],
            "epochs": a.epochs,
            "alpha_floor": a.alpha_floor,
            "final_loss": round(history[-1][2], 5),
            "per_regime": summary,
        }
        export_gate(agent.head, a.out, provenance)
        print(f"saved {a.out}")
        stem = os.path.splitext(os.path.basename(a.out))[0]
        plot(history, handoff(agent.head), os.path.join(ROOT, "results", f"{stem}_training.png"))
