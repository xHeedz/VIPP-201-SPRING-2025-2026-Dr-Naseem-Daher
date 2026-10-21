"""
Animated replay (GIF) of one driver as the labeling agent sees it, second by second: speed, gap to the car ahead,
the state the agent receives, its action probabilities (PPO) or action values (DQN), and the decision.

    python scripts/make_explainer_video.py --config small6 [--algo ppo] [--pick normal_correct_late] [--out PATH]

Uses the agent saved by scripts/state_reward_ablation.py (data/ablation_<algo>_<config>.zip) and the test vehicles
of the same split. --pick chooses the vehicle: <true class>_<correct|wrong>_<early|late> (median decision time of
that group decides early / late), or an index into the test list. Also writes the second by second trace as CSV
(<out>.csv) so every frame can be checked against the numbers.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR  # noqa: E402

LABELS = {"pts_speed_run": "speed points (so far)", "pts_accel2_run": "accel² points (so far)",
          "pts_prox2_run": "gap points (so far)", "pts_wave_run": "lane points (so far)",
          "pts_speed_rec": "speed points (last 10 s)", "pts_accel2_rec": "accel² points (last 10 s)",
          "pts_prox2_rec": "gap points (last 10 s)", "pts_wave_rec": "lane points (last 10 s)",
          "score_run": "score so far", "score_rec": "score, last 10 s", "prox_follow": "gap points while following",
          "leader_share": "share of seconds with a car ahead", "time": "time watched (of 60 s)",
          "env_highway": "highway", "env_urban": "urban", "env_weather": "weather"}
COL = {"wait": "#B9AEB0", "wait 5 s": "#D9CFD1", "conservative": "#4A6FA5", "normal": "#D9A441", "aggressive": "#7A1F2B"}


def trace(model, env, i, algo):
    import torch
    obs, _ = env.reset(options={"seq": i, "t0": 0})
    s = env.sequences[i]
    names = [("wait" if k == 1 else f"wait {k} s") for k in env.waits] + list(env.classes)
    rows, done = [], False
    while not done:
        o = torch.as_tensor(obs[None])
        with torch.no_grad():
            if algo == "ppo":
                vals = model.policy.get_distribution(o).distribution.probs[0].numpy()
            else:
                vals = model.q_net(o)[0].numpy()
        a, _ = model.predict(obs, deterministic=True)
        n = env.n
        rows.append({"second": n, "action": names[int(a)], **{f: float(v) for f, v in zip(env.features, obs)},
                     **{f"value_{k}": float(v) for k, v in zip(names, vals)}})
        obs, _, done, _, info = env.step(int(a))
    return pd.DataFrame(rows), s, info, names


def choose(episodes, pick):
    if pick.isdigit():
        return int(pick)
    cls, ok, when = pick.split("_")
    e = episodes.reset_index().rename(columns={"index": "i"})
    g = e[(e["true"] == cls) & ((e["true"] == e["pred"]) == (ok == "correct"))]
    med = g["seconds"].median()
    g = g[g["seconds"] >= med] if when == "late" else g[g["seconds"] <= med]
    return int(g.iloc[(g["seconds"] - (g["seconds"].quantile(0.6 if when == "late" else 0.4))).abs().argsort().iloc[0]]["i"])


def render(tr, seq, info, names, features, out, algo, fps=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    T = int(tr["second"].max())                 # seconds watched (one frame per decision; "wait 5 s" covers 5)
    speed = seq["speed_kmh"][:T]
    gap = np.nan_to_num(seq["gap_m"][:T], nan=0.0)
    lead = seq["leader"][:T]
    fig = plt.figure(figsize=(12, 6.75), dpi=90)
    fig.patch.set_facecolor("#FBF8F6")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45, wspace=0.35, left=0.24, right=0.97, top=0.86, bottom=0.08)
    ax_s, ax_g, ax_x, ax_p = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
                              fig.add_subplot(gs[1, 1]))
    title = fig.text(0.5, 0.94, "", ha="center", fontsize=17, color="#231A1C", weight="bold")
    sub = fig.text(0.5, 0.895, f"true type: {seq['label']} ({seq['environment']}); the agent does not know it", ha="center",
                   fontsize=11, color="#8E8385")
    vkey = "probability" if algo == "ppo" else "value (expected reward)"
    frames = list(range(len(tr))) + [len(tr) - 1] * 6

    def draw(f):
        n = int(tr.iloc[f]["second"])
        for ax in (ax_s, ax_g, ax_x, ax_p):
            ax.cla()
            ax.set_facecolor("white")
        x = np.arange(1, n + 1)
        ax_s.plot(x, speed[:n], color="#7A1F2B", lw=2.2)
        ax_s.set_xlim(1, max(T, 10))
        ax_s.set_ylim(0, max(130, np.nanmax(speed) * 1.1))
        ax_s.set_title("speed (km/h)", fontsize=12, loc="left")
        ax_g.bar(x, np.where(lead[:n] == 1, np.minimum(gap[:n], 100), 100), width=0.9,
                 color=np.where(lead[:n] == 1, "#D9A441", "#F1ECEC"))
        ax_g.set_xlim(0.5, max(T, 10) + 0.5)
        ax_g.set_ylim(0, 100)
        ax_g.set_title("gap to the car ahead (m), gold = car ahead", fontsize=12, loc="left")
        r = tr.iloc[f]
        vals = [r[c] for c in features]
        ax_x.barh(range(len(features)), vals, color="#7A1F2B")
        ax_x.set_yticks(range(len(features)))
        ax_x.set_yticklabels([LABELS.get(c, c) for c in features], fontsize=10)
        ax_x.invert_yaxis()
        ax_x.set_xlim(min(0, min(vals) * 1.1), max(1.0, max(vals) * 1.1))
        ax_x.set_title("what the agent sees (its state)", fontsize=12, loc="left")
        pv = [r[f"value_{k}"] for k in names]
        ax_p.bar(range(len(names)), pv, color=[COL.get(k, "#999") for k in names])
        ax_p.set_xticks(range(len(names)))
        short = {"wait 5 s": "wait 5 s", "conservative": "conserv.", "aggressive": "aggress."}
        ax_p.set_xticklabels([short.get(k, k) for k in names], fontsize=10)
        if algo == "ppo":
            ax_p.set_ylim(0, 1)
        ax_p.set_title(f"each button's {vkey}", fontsize=12, loc="left")
        if f < len(tr) - 1 or r["action"].startswith("wait"):
            title.set_text(f"second {int(r['second'])}: the agent presses '{r['action']}'")
        else:
            ok = "correct" if info.get("predicted") == seq["label"] else "wrong"
            title.set_text(f"second {int(r['second'])}: labels {info.get('predicted', 'none').upper()} ({ok})")
        for ax in (ax_s, ax_g, ax_x, ax_p):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main(config, algo, pick, out):
    from stable_baselines3 import DQN, PPO
    from state_reward_ablation import CLASSES, CONFIGS, TEST_SEEDS
    from env.labeling_ref import LabelingRefEnv, sequences_from_clean
    cfg = CONFIGS[config]
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted.csv.gz"))
    d = d[d["group"].isin(TEST_SEEDS)]
    test = [s for s in sequences_from_clean(d) if len(s["terms"]) >= 60]
    episodes = pd.read_csv(os.path.join(DATA_DIR, f"ablation_episodes_{algo}_{config}.csv.gz"))
    i = choose(episodes, pick)
    model = (PPO if algo == "ppo" else DQN).load(os.path.join(DATA_DIR, f"ablation_{algo}_{config}.zip"))
    env = LabelingRefEnv(test, CLASSES, **cfg)
    tr, seq, info, names = trace(model, env, i, algo)
    tr.to_csv(out.replace(".gif", ".csv"), index=False)
    render(tr, seq, info, names, cfg["features"], out, algo)
    print(f"vehicle {seq['trip']} ({seq['label']}), {int(tr['second'].max())} s in {len(tr)} decisions, decision {info.get('predicted')}; written {out}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="small6")
    p.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--pick", default="normal_correct_late")
    p.add_argument("--out", default=os.path.join(DATA_DIR, "explainer.gif"))
    a = p.parse_args()
    main(a.config, a.algo, a.pick, a.out)
