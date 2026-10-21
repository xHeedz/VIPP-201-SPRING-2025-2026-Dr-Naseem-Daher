"""
Seven slide weekly update (week of 18 Oct 2026): why each model reaches the accuracy it reaches, in plain words.
Same style as the October deck (helpers from scripts/build_october_deck.py).
Write-up: docs/research_notebook/fall_2026_10_18_why_the_numbers.md.

    python scripts/build_diagnosis_deck.py
writes ../18-10-2026/Weekly Update.pptx
"""
import glob
import math
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_october_deck as B  # noqa: E402
from build_october_deck import (GREY_BAR, INK, MAROON, MAROON2, MUTED, PALE, PINK, W, WINE, X0,  # noqa: E402
                                bullet_lines, card, chart_card, color_series, header, new_slide, stat,
                                style_chart, text)
from model.aggressiveness_model import original_score  # noqa: E402
from model.irl import A_GRID, decisions  # noqa: E402

DATA, ROOT = B.DATA, B.ROOT
OUT_DIR = os.path.abspath(os.path.join(ROOT, "..", "18-10-2026"))
OUT = os.path.join(OUT_DIR, "Weekly Update.pptx")
GOLD, BLUE = "D9A441", "4A6FA5"
B.TOTAL = 7


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bars(ch, top, unit, fmt='0"%"', labels='0"%"'):
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = top
    ch.value_axis.major_unit = unit
    ch.value_axis.tick_labels.number_format = fmt
    ch.value_axis.tick_labels.number_format_is_linked = False
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.has_data_labels = True
    pl.data_labels.number_format = labels
    pl.data_labels.number_format_is_linked = False
    pl.data_labels.font.size = Pt(11)


# ── data ────────────────────────────────────────────────────────────────────
def uah_numbers():
    w = pd.read_csv(os.path.join(DATA, "uah_windows.csv"))
    w = w[w.behavior.isin(["normal", "aggressive"])].copy()
    w["score"] = original_score(w[["phi_speed", "phi_accel", "phi_prox", "phi_wave"]].to_numpy())
    by_t = {}
    for name, k in [("10 s", 1), ("1 min", 11), ("2 min", 23), ("whole trip", None)]:
        res = []
        for _, g in w.groupby("driver"):
            parts = []
            for _, t in g.groupby("trip"):
                s = t.sort_values("t0").score.to_numpy()
                m = [s.mean()] if k is None else [s[i:i + k].mean() for i in range(0, max(len(s) - k + 1, 1), k)]
                parts += [(t.behavior.iloc[0], x) for x in m]
            p = pd.DataFrame(parts, columns=["b", "s"])
            res.append(auc(p[p.b == "aggressive"].s, p[p.b == "normal"].s))
        by_t[name] = float(np.mean(res))
    per_driver = {d: auc(g[g.behavior == "aggressive"].score, g[g.behavior == "normal"].score) for d, g in w.groupby("driver")}
    lodo = pd.read_csv(os.path.join(DATA, "uah_lodo_results.csv"))
    rel = pd.read_csv(os.path.join(DATA, "uah_relabel.csv"))
    rel = rel[rel.proximity == "metres"].groupby("labels").agent_auc.mean()
    return dict(by_t=by_t, d1=per_driver["D1"], others=float(np.mean([v for d, v in per_driver.items() if d != "D1"])),
                formula_auc=float(lodo.original_auc.mean()), agent_auc=float(lodo.agent_auc.mean()),
                agent_acc=float(lodo.agent_window_acc.mean()), relabel_gain=float(rel["window"] - rel["trip"]))


def planted_numbers():
    q = pd.read_csv(os.path.join(DATA, "labeling_rl_planted_episodes_headway_wb4_seed0.csv.gz"))
    q = q[q.method == "rl_ppo"]
    ppo_normal = float((q[q.true == "normal"].pred == "normal").mean())
    hw = [pd.read_csv(os.path.join(DATA, f"labeling_rl_planted_summary_headway_wb4_seed{k}.csv"), index_col=0) for k in (0, 1, 2)]
    m = pd.read_csv(os.path.join(DATA, "labeling_rl_planted_summary_seed0.csv"), index_col=0)
    # speedFactor 0.9 / 1.0 / 1.2 with speedDev 0.1 (scripts/planted_drivers.py), best boundaries 0.95 and 1.10
    ceiling = np.mean([phi(0.5), phi(1.0) - phi(-0.5), phi(1.0)])
    return dict(ceiling=float(ceiling), formula=float(m.loc["refit_env_T60", "balanced_acc"]),
                formula_gap_s=float(hw[0].loc["refit_env_T60", "balanced_acc"]),
                ppo=float(np.mean([d.loc["rl_ppo", "balanced_acc"] for d in hw])),
                ppo_s=float(np.mean([d.loc["rl_ppo", "seconds"] for d in hw])), ppo_normal=ppo_normal)


def irl_numbers():
    zero = {}
    for scen in ["highway_low", "highway_medium", "jam_jam"]:
        ch = []
        for f in glob.glob(os.path.join(DATA, "sumo_planted", f"{scen}_s[789].per_second.csv.gz")):
            for _, g in pd.read_csv(f).groupby("vehicle_id"):
                _, y = decisions(g.sort_values("t"))
                ch += list(A_GRID[y])
        zero[scen] = float((pd.Series(ch) == 0).mean())
    v = pd.read_csv(os.path.join(DATA, "irl_planted_summary.csv")).set_index("comparison").loc["aggressive vs normal"]
    pop = pd.read_csv(os.path.join(DATA, "irl_population.csv")).set_index("group").desired_headway_s
    return dict(zero=zero, irl_auc=float(v["auc_irl_combined"]), formula_auc=float(v["auc_index_mix"]), pop=pop)


# ── slides ──────────────────────────────────────────────────────────────────
def slide_models(prs, crest, n):
    s = new_slide(prs)
    header(s, "1 · Overview", "Four models, three kinds of data", crest)
    models = [("The formula", "fixed weights on speed, acceleration, gap, lane offset", "learns: no (baseline)"),
              ("Supervised", "same formula, learns its own weights and cut off from labels", "learns: yes"),
              ("PPO (RL)", "watches a driver each second, decides when to label", "learns: yes"),
              ("Inverse RL", "works out what a driver wants, e.g. preferred gap", "learns: yes")]
    w = (W - 3 * 0.25) / 4
    for i, (t, b, l) in enumerate(models):
        x = X0 + i * (w + 0.25)
        card(s, x, 1.65, w, 1.95, fill=MAROON if i == 0 else B.CARD)
        dark = i == 0
        text(s, x + 0.2, 1.8, w - 0.4, 0.35, [[(t, {"bold": True, "size": 16, "color": "FFFFFF" if dark else MAROON2})]])
        text(s, x + 0.2, 2.25, w - 0.4, 0.85, b, size=12, color=PALE if dark else INK)
        text(s, x + 0.2, 3.15, w - 0.4, 0.3, l, size=11, color=PINK if dark else MUTED)
    data = [("UAH", "6 real drivers, told to drive normally or aggressively; labelled per trip",
             "train on 5 drivers, test on the 6th, 6 times"),
            ("Simulated (SUMO)", "drivers planted as conservative, normal or aggressive",
             "train on runs 0 to 6, test on runs 7 to 9"),
            ("German and US traffic", "highD, exiD, inD, rounD, uniD, NGSIM: real traffic",
             "no labels: no accuracy possible, only shares")]
    w = (W - 2 * 0.25) / 3
    for i, (t, b, l) in enumerate(data):
        x = X0 + i * (w + 0.25)
        card(s, x, 3.85, w, 1.85, fill="FFFFFF", line=B.CARD_LINE)
        text(s, x + 0.2, 4.0, w - 0.4, 0.35, [[(t, {"bold": True, "size": 15, "color": MAROON2})]])
        text(s, x + 0.2, 4.42, w - 0.4, 0.7, b, size=12, color=INK)
        text(s, x + 0.2, 5.2, w - 0.4, 0.4, l, size=11, color=MUTED)
    text(s, X0, 5.9, W, 0.6, [[("Accuracy", {"bold": True, "color": MAROON2}), (": share of drivers labelled right.   ", {}),
                               ("AUC", {"bold": True, "color": MAROON2}),
                               (": chance an aggressive driver scores higher than a normal one (0.5 coin flip, 1.0 perfect).", {})]],
         size=12, color=INK)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = ("The formula is not trained; every other model is compared against it. "
                                           "German and US traffic has no aggressiveness labels, so only shares can be reported there.")


def slide_uah(prs, crest, n, u):
    s = new_slide(prs)
    header(s, "3 · Real drivers (UAH)", "The labels are the limit, not the model", crest)
    cd = CategoryChartData()
    cd.categories = list(u["by_t"])
    cd.add_series("AUC", [round(v, 2) for v in u["by_t"].values()])
    chart_card(s, X0, 1.65, 6.6, 4.92, "The formula, scored on longer and longer stretches of driving (AUC)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.3), Inches(4.4), cd).chart
    style_chart(ch, legend=False)
    color_series(ch, [MAROON2])
    bars(ch, 1.0, 0.25, fmt="0.00", labels="0.00")
    x2, w2 = X0 + 6.85, W - 6.85
    stat(s, x2, 1.65, w2, f"{u['formula_auc']:.2f}", f" / {u['agent_auc']:.2f}",
         f"AUC on 10 s clips: formula / supervised (supervised accuracy {100 * u['agent_acc']:.0f}%)", dark=True)
    card(s, x2, 2.75, w2, 3.82)
    text(s, x2 + 0.25, 2.9, w2 - 0.5, 0.3, [[("Why", {"bold": True, "color": MAROON2, "size": 15})]])
    text(s, x2 + 0.25, 3.3, w2 - 0.5, 3.2, [
        "All of the missing 0.06 comes from one trip. On whole trips, five of the six drivers are perfect (1.00): every "
        "aggressive trip scores higher than every normal trip of the same driver. Driver D1 scores 0.67 because D1's "
        "aggressive trip on the secondary road scores 40.9, lower than two of D1's normal trips (45.5 and 41.0). The "
        "average of 1, 1, 1, 1, 1 and 0.67 is 0.94.",
        "The reason for that one trip: the phone camera detected a car ahead only 10% of the time on it (59% on D1's "
        "normal trips). The gap term, the heaviest in the formula (weight 0.8), read zero almost the whole trip. That is a "
        "sensor problem, not the driving."], size=11, color=INK)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Like grading every sentence of an essay with the essay's grade. Averaging over a whole trip washes out the calm "
        "moments, which is why the same formula gets 93% on trips and 77% on 10 s clips. Only 6 drivers, so small "
        "differences between methods are noise.")


def slide_sim_ppo(prs, crest, n, p):
    s = new_slide(prs)
    header(s, "4 · Simulated drivers and PPO", "The driver types overlap, and the models already hit that limit", crest)
    # preferred speed of each type: speedFactor 0.9 / 1.0 / 1.2, speedDev 0.1 (scripts/planted_drivers.py)
    xs = list(range(60, 152, 2))
    cd = CategoryChartData()
    cd.categories = [f"{x}%" for x in xs]
    for name, mu in [("conservative", 90), ("normal", 100), ("aggressive", 120)]:
        cd.add_series(name, [round(math.exp(-0.5 * ((x - mu) / 10) ** 2), 3) for x in xs])
    chart_card(s, X0, 1.65, 6.6, 4.92, "Preferred speed of each simulated car (% of the speed limit)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(X0 + 0.15), Inches(2.05), Inches(6.3), Inches(4.4), cd).chart
    style_chart(ch)
    for ser, c in zip(ch.plots[0].series, [BLUE, GOLD, MAROON2]):
        ser.format.line.color.rgb = B.rgb(c)
        ser.format.line.width = Pt(3)
        ser.smooth = True
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    from pptx.oxml.ns import qn
    cat = ch.category_axis._element
    for tag, val in (("c:tickLblSkip", "5"), ("c:tickMarkSkip", "5")):
        el = cat.find(qn(tag))
        if el is None:
            el = cat.makeelement(qn(tag), {})
            cat.append(el)
        el.set("val", val)
    text(s, X0 + 0.3, 6.12, 6.0, 0.3, "where two curves overlap, cars of different types drive the same speed",
         size=11, color=MUTED)
    x2, w2 = X0 + 6.85, W - 6.85
    card(s, x2, 1.65, w2, 1.55, fill=MAROON)
    text(s, x2 + 0.25, 1.8, w2 - 0.5, 1.3, [
        [("The setup", {"bold": True, "size": 14, "color": "FFFFFF"})],
        [("each car gets its own preferred speed: conservative about 90%, normal 100%, aggressive 120%, "
          "give or take 10%", {"color": PALE})]], size=12)
    card(s, x2, 3.4, w2, 1.75, fill="FFFFFF", line=B.CARD_LINE)
    rows = [("best possible from speed", f"{100 * p['ceiling']:.0f}%"), ("formula, 60 s", f"{100 * p['formula']:.0f}%"),
            ("formula + gap in seconds, 60 s", f"{100 * p['formula_gap_s']:.0f}%"),
            (f"PPO, {p['ppo_s']:.0f} s", f"{100 * p['ppo']:.0f}%")]
    text(s, x2 + 0.25, 3.5, w2 - 1.3, 1.6, [[(r[0], {})] for r in rows], size=12, color=INK, line=1.25)
    text(s, x2 + w2 - 1.05, 3.5, 0.8, 1.6, [[(r[1], {"bold": True, "color": MAROON2})] for r in rows], size=12,
         align=PP_ALIGN.RIGHT, line=1.25)
    card(s, x2, 5.35, w2, 1.22)
    text(s, x2 + 0.25, 5.45, w2 - 0.5, 1.05, [
        f"Normal is hardest ({100 * p['ppo_normal']:.0f}% right): it overlaps on both sides.",
        "PPO sees the same data, so it gets the same 74%, just in half the time."], size=11, color=INK)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Best possible from speed: with these overlaps no method can do better than 69% using speed. The gap in seconds "
        "adds a little (each type keeps a different following distance), but only while following another car.")


def slide_irl(prs, crest, n, r):
    s = new_slide(prs)
    header(s, "5 · Inverse RL", "Most seconds of driving teach it nothing", crest)
    names = {"highway_low": "highway, light", "highway_medium": "highway, medium", "jam_jam": "jam"}
    cd = CategoryChartData()
    cd.categories = [names[k] for k in r["zero"]]
    cd.add_series("seconds spent just keeping the same speed", [round(100 * v, 0) for v in r["zero"].values()])
    chart_card(s, X0, 1.65, 6.6, 4.92, "Seconds where the driver just keeps the same speed")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.3), Inches(4.4), cd).chart
    style_chart(ch, legend=False)
    color_series(ch, [MAROON2])
    bars(ch, 100, 20, labels='0"%"')
    pop = r["pop"]
    x2, w2 = X0 + 6.85, W - 6.85
    stat(s, x2, 1.65, w2, f"{r['irl_auc']:.2f}", f" / {r['formula_auc']:.2f}", "AUC per car: inverse RL / formula", dark=True)
    stat(s, x2, 2.75, w2, f"{pop['planted aggressive']:.1f}", f" / {pop['planted conservative']:.1f} s",
         "preferred gap found when all cars of a type are pooled: aggressive / conservative")
    card(s, x2, 3.85, w2, 2.72)
    text(s, x2 + 0.25, 4.0, w2 - 0.5, 0.3, [[("Why", {"bold": True, "color": MAROON2, "size": 15})]])
    bullet_lines(s, x2 + 0.25, 4.45, w2 - 0.5, [
        "it learns from reactions: braking, speeding up, closing a gap",
        "one car gives only a handful of reactions",
        "pooled over many cars it works; per car there is not enough"], step=0.68, size=12)
    B.footer(s, n)


def slide_next(prs, crest, n):
    s = new_slide(prs)
    header(s, "7 · Next", "What could move the numbers", crest)
    items = [("Better labels", ["clip level labels instead of trip labels", "more drivers (full DriveDNA, 465 drivers)"]),
             ("Better inputs for PPO", ["gap in seconds, not only metres", "is there a car ahead", "traffic speed around the car"]),
             ("Richer reward", ["different cost per mistake type", "waiting cost that grows over time",
                                "penalty for confident early wrong guesses"]),
             ("DQN comparison", ["designed for small discrete action sets like this one", "same library, easy to add",
                                 "expect similar accuracy (same inputs)"])]
    w = (W - 3 * 0.25) / 4
    for i, (t, its) in enumerate(items):
        x = X0 + i * (w + 0.25)
        card(s, x, 1.65, w, 3.35, fill=B.CARD)
        text(s, x + 0.2, 1.8, w - 0.4, 0.35, [[(t, {"bold": True, "size": 15, "color": MAROON2})]])
        bullet_lines(s, x + 0.2, 2.3, w - 0.4, its, step=0.85, size=12)
    card(s, X0, 5.2, W, 1.35, fill=MAROON)
    text(s, X0 + 0.3, 5.32, W - 0.6, 0.3, [[("To decide with Dr. Daher", {"bold": True, "color": "FFFFFF", "size": 15})]])
    text(s, X0 + 0.3, 5.72, W - 0.6, 0.75, ["Which first: labels, inputs or reward?   Widen the gap between simulated "
                                            "driver types, or keep the realistic overlap?   Inverse RL: fit only on the "
                                            "seconds where a driver reacts?"], size=12, color=PALE)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "A richer reward cannot raise the ceiling set by labels and inputs, but it can change how PPO spends its "
        "information. PPO already works with discrete actions; DQN is a fair comparison, not a fix.")


def formula_card(s, x, y, w, h, title, lines, dark=False):
    card(s, x, y, w, h, fill=MAROON if dark else B.CARD)
    text(s, x + 0.22, y + 0.14, w - 0.44, 0.32, [[(title, {"bold": True, "size": 15, "color": "FFFFFF" if dark else MAROON2})]])
    paras = []
    for ln in lines:
        if isinstance(ln, tuple):      # (equation, explanation)
            paras.append([(ln[0], {"bold": True, "color": "FFFFFF" if dark else INK, "size": 13})])
            paras.append([(ln[1], {"color": PALE if dark else MUTED, "size": 11})])
        else:
            paras.append([(ln, {"color": PALE if dark else INK, "size": 12})])
    text(s, x + 0.22, y + 0.55, w - 0.44, h - 0.65, paras, size=12)


def slide_formulas(prs, crest, n):
    s = new_slide(prs)
    header(s, "2 · The maths", "What each model computes and what it learns from", crest)
    w, h = (W - 0.25) / 2, 2.4
    formula_card(s, X0, 1.6, w, h, "The formula (no learning)", [
        ("AI = 100 × (0.5·speed² + 0.2·accel + 0.8·gap² + 0.4·lane)",
         "speed = km/h ÷ 150, accel = size of acceleration ÷ 5, gap = 1 − metres ÷ 50, lane = offset ÷ 1.5"),
        ("label: below 29 conservative, below 42 normal, else aggressive", "cut offs fitted on labelled data")], dark=True)
    formula_card(s, X0 + w + 0.25, 1.6, w, h, "Supervised", [
        ("score = 100 × (w1·speed² + w2·accel + w3·gap² + w4·lane)", "w1 to w4 learned per environment, they add up to 1"),
        ("P(aggressive) = sigmoid((score − threshold) ÷ τ)",
         "The sigmoid turns \"how far above the threshold\" into a probability between 0 and 1. "
         "The threshold and τ (how sharp the switch is) are learned too."),
        ("loss = cross entropy: −log P(true label)", "both classes weighted equally")])
    formula_card(s, X0, 4.2, W, h, "PPO (RL)", [
        ("reward: wait −0.002 per second, correct +1, wrong −1",
         "on UAH a missed aggressive driver costs −2; no answer after 60 to 120 s counts as wrong"),
        ("goal: largest total reward per driver", "")])
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Supervised: model/dynamic_weight_agent.py fit_labels. PPO reward: env/labeling_env.py step. "
        "The supervised loss makes the true label as likely as possible; PPO gets no right answer per second, "
        "only a reward at the end.")


def slide_dqn(prs, crest, n, dq):
    s = new_slide(prs)
    header(s, "6 · Next model: DQN", "Same task, a learner made for few discrete actions", crest)
    w = 5.55
    formula_card(s, X0, 1.6, w, 2.55, "What DQN is", [
        ("Q(s, a) = total reward expected if action a is taken now",
         "one number for each of the 4 actions: wait, conservative, normal, aggressive"),
        ("pick the action with the largest Q", "learn: Q(s, a) moves toward r + largest Q of the next second")], dark=True)
    formula_card(s, X0, 4.35, w, 2.2, "PPO against DQN", [
        "PPO learns the probability of each action directly",
        "DQN learns the value of each action, then picks the best",
        "both work with discrete actions; DQN is the classic choice for 4 actions",
        "same inputs, so a similar ceiling is expected"])
    x2, w2 = X0 + w + 0.25, W - w - 0.25
    stat(s, x2, 1.6, w2, dq["acc"], "", dq["sub"])
    card(s, x2, 2.75, w2, 3.8, fill="FFFFFF", line=B.CARD_LINE)
    text(s, x2 + 0.25, 2.9, w2 - 0.5, 0.3, [[("Plan, step by step", {"bold": True, "color": MAROON2, "size": 15})]])
    steps = [("[x]", "script: same environment, reward, drivers and test as PPO"),
             ("[x]", "short trial run (results above, not final)"),
             ("[ ]", "check early guesses: random exploration ends 3 of 4 first episodes (trial: 10% decided after 1 s)"),
             ("[ ]", "full training, 3 seeds, same features as the best PPO"),
             ("[ ]", "compare PPO, DQN and the formula on the same test drivers"),
             ("[ ]", "repeat on UAH (train on 5 drivers, test on the 6th)")]
    text(s, x2 + 0.25, 3.35, w2 - 0.5, 3.1, [[(a + "  ", {"bold": True, "color": MAROON2}), (b, {})] for a, b in steps],
         size=12, color=INK)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = "Script: scripts/train_labeling_dqn_planted.py (reuses train_labeling_rl_planted.py)."


def dqn_numbers():
    f = os.path.join(DATA, "labeling_dqn_planted_summary_trial_headway_seed0.csv")
    if not os.path.exists(f):
        return dict(acc="…", sub="first trial run in progress")
    d = pd.read_csv(f, index_col=0).loc["rl_dqn"]
    return dict(acc=f"{100 * d['balanced_acc']:.0f}%", sub=f"first trial: accuracy after {d['seconds']:.0f} s on average "
                f"(PPO 74%), short training, not final")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    u, p, r = uah_numbers(), planted_numbers(), irl_numbers()
    prs = Presentation(B.SRC)
    with zipfile.ZipFile(B.SRC) as z:
        crest = os.path.join(tempfile.mkdtemp(), "crest.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):                      # start from the September deck's master, without its slides
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    slide_models(prs, crest, 1)
    slide_formulas(prs, crest, 2)
    slide_uah(prs, crest, 3, u)
    slide_sim_ppo(prs, crest, 4, p)
    slide_irl(prs, crest, 5, r)
    slide_dqn(prs, crest, 6, dqn_numbers())
    slide_next(prs, crest, 7)
    assert len(prs.slides) == B.TOTAL, len(prs.slides)
    prs.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    main()
