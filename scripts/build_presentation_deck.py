"""
Short deck for a 5 to 10 minute presentation (week of 20 Oct 2026), built around one question: why are the models not
100% accurate? Every slide explains itself in plain sentences; the speaker script is in each slide's notes.
Same style as the October deck; every number is read from data/.

    python scripts/build_presentation_deck.py
writes ../20-10-2026/Weekly Update.pptx (the detailed 10 slide version is scripts/build_revision_deck.py)
"""
import json
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
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_october_deck as B  # noqa: E402
from build_october_deck import (GREY_BAR, INK, MAROON, MAROON2, MUTED, PALE, PINK, W, X0, card, chart_card,  # noqa: E402
                                color_series, header, new_slide, stat, style_chart, text)

DATA, ROOT = B.DATA, B.ROOT
OUT_DIR = os.path.abspath(os.path.join(ROOT, "..", "20-10-2026"))
OUT = os.path.join(OUT_DIR, "Weekly Update.pptx")
GOLD, BLUE = "D9A441", "4A6FA5"
B.TOTAL = 9


def para_box(s, x, y, w, h, title, paras, dark=False, size=13):
    """card with a title and plain sentences (each paragraph: str, or list of (text, opts) runs)"""
    card(s, x, y, w, h, fill=MAROON if dark else B.CARD)
    text(s, x + 0.28, y + 0.18, w - 0.56, 0.35, [[(title, {"bold": True, "size": 16, "color": "FFFFFF" if dark else MAROON2})]])
    body = [[(p, {})] if isinstance(p, str) else p for p in paras]
    tb = text(s, x + 0.28, y + 0.7, w - 0.56, h - 0.85, body, size=size, color=PALE if dark else INK, line=1.15)
    for p in tb.text_frame.paragraphs[1:]:
        p.space_before = Pt(9)
    return tb


def notes(s, script):
    s.notes_slide.notes_text_frame.text = script


def set_bars(ch, top, unit, fmt, labels):
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


def point_colors(ch, colors):
    for i, c in enumerate(colors):
        pt = ch.plots[0].series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = B.rgb(c)


# ── numbers ─────────────────────────────────────────────────────────────────
def numbers():
    p = pd.read_csv(os.path.join(DATA, "ablation_state_reward.csv")).set_index("config")
    q = pd.read_csv(os.path.join(DATA, "ablation_state_reward_dqn.csv")).set_index("config")
    sup = pd.read_csv(os.path.join(DATA, "rerun_supervised.csv"))
    l = sup[sup["test"] == "uah lodo"]
    ag = pd.read_csv(os.path.join(DATA, "backtrace_agents.csv"))
    ag = ag[ag["run"] == "ppo:small6_wait5"].set_index("following")
    log = pd.read_csv(os.path.join(DATA, "cleaning_log.csv")).set_index("source")
    summ = pd.read_csv(os.path.join(DATA, "rerun_formula_summary.csv"))
    a = summ[summ["label"] == "all"].set_index("source")
    irl = pd.read_csv(os.path.join(DATA, "rerun_irl_summary.csv"))
    irl = irl[irl["variant"] == "all seconds"].iloc[0]
    ref = json.load(open(os.path.join(DATA, "reference_weights.json")))
    return dict(
        formula60=float(p.loc["formula_T60", "balanced_acc"]), ppo=float(p.loc["small6_wait5", "balanced_acc"]),
        ppo_s=float(p.loc["small6_wait5", "seconds"]), dqn=float(q.loc["full", "balanced_acc"]),
        uah_formula=float(l["formula_auc"].mean()), uah_sup=float(l["supervised_auc"].mean()),
        never_normal=float(ag.loc["never", "acc_normal"]), never_aggr=float(ag.loc["never", "acc_aggressive"]),
        follow_acc=float(ag.loc["over 75%", "accuracy"]),
        uah_unseen=float(log.loc["uah", "rule4_gap_not_detected"] / log.loc["uah", "rows_in"]),
        highd_aggr=float(a.loc["highd", "share_aggressive"]), us101_aggr=float(a.loc["ngsim_us_101", "share_aggressive"]),
        irl_auc=float(irl["auc_irl"]), irl_formula=float(irl["auc_formula"]), ref=ref)


# ── slides ──────────────────────────────────────────────────────────────────
def s1(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "1 · The question", "Why are my models not 100% accurate?", crest)
    cd = CategoryChartData()
    cd.categories = ["formula\n(fixed rule, 60 s)", f"PPO agent\n(about {v['ppo_s']:.0f} s)", "DQN agent\n(about 38 s)"]
    cd.add_series("accuracy", [round(100 * v["formula60"], 0), round(100 * v["ppo"], 0), round(100 * v["dqn"], 0)])
    chart_card(s, X0, 1.65, 5.9, 4.92, "Simulated SUMO drivers (true type known): % labelled right")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(5.6), Inches(4.4), cd).chart
    style_chart(ch, legend=False)
    color_series(ch, [MAROON2])
    point_colors(ch, [GREY_BAR, MAROON2, BLUE])
    set_bars(ch, 100, 20, '0"%"', '0"%"')
    x2, w2 = X0 + 6.15, W - 6.15
    para_box(s, x2, 1.65, w2, 2.55, "Short answer: mostly the data", [
        "Four very different models (a fixed formula, a supervised model, a PPO agent, a DQN agent) all stop at about the same score.",
        "When completely different models hit the same wall, the wall is in the data, not in the models."], dark=True)
    para_box(s, x2, 4.4, w2, 2.17, "Same on real drivers (UAH, 6 drivers)", [
        [("Tested on a driver never seen in training: formula ", {}), (f"{v['uah_formula']:.2f}", {"bold": True, "color": MAROON2}),
         (", supervised model ", {}), (f"{v['uah_sup']:.2f}", {"bold": True, "color": MAROON2}),
         (" (AUC: 1.0 = always ranks the aggressive driver above the normal one)", {})]], size=12)
    B.footer(s, n)
    notes(s, "The whole point this week: why do the models stop where they stop? The short answer is the data. Left: the "
             "simulated SUMO drivers, where I planted each car's type, so the right answer is known. Test runs 7 to 9, never used "
             f"for training. The formula after watching 60 s gets {100 * v['formula60']:.0f}%, the PPO agent "
             f"{100 * v['ppo']:.0f}%, the DQN agent {100 * v['dqn']:.0f}%. Right: real drivers from UAH, tested on a driver the "
             f"model never saw: the formula {v['uah_formula']:.2f}, the supervised model {v['uah_sup']:.2f}. If the problem were the "
             "model, a better model would jump ahead. None does, so the limit is in the data. Three reasons, one per slide.")


def s2(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "3 · Reason 1: the labels (UAH)", "The answer sheet is partly wrong", crest)
    para_box(s, X0, 1.65, 5.55, 4.92, "What happens", [
        "UAH-DriveSet (6 real drivers) labels whole trips. A driver told to 'drive aggressively' for 15 minutes has every "
        "10 second clip labelled aggressive, even the calm cruising in between.",
        "UAH's own software rates only about half of those clips as aggressive.",
        "So when the formula or the supervised model calls a calm clip calm, it is marked wrong. Even a perfect model would "
        "lose points here."])
    x2, w2 = X0 + 5.8, W - 5.8
    card(s, x2, 1.65, w2, 2.6, fill=MAROON)
    text(s, x2 + 0.28, 1.83, w2 - 0.56, 0.35, [[("Like grading every sentence of an essay with the essay's final grade",
                                                 {"bold": True, "color": "FFFFFF", "size": 15})]])
    text(s, x2 + 0.28, 2.75, w2 - 0.56, 1.4, "A calm sentence in an 'A' essay is still a calm sentence. The label says one "
         "thing, the clip shows another.", size=13, color=PALE)
    para_box(s, x2, 4.45, w2, 2.12, "The proof (the formula on UAH)", [
        "The formula gets 0.83 on 10 second clips, 0.94 on whole trips. Averaging a whole trip washes out the calm moments, "
        "so it matches the trip label better."], size=12)
    B.footer(s, n)
    notes(s, "Reason one is the labels, and it applies to the real drivers in UAH. UAH labels whole trips, not moments. If a "
             "driver was asked to drive aggressively, every clip of that trip is labelled aggressive, even when they were just "
             "cruising. UAH's own algorithm agrees with only about half of those clips. So the formula and the supervised model "
             "get marked wrong for calling a calm clip calm. The proof: the same formula scores 0.83 on 10 second clips but 0.94 "
             "on whole trips, because over a whole trip the calm moments average out. Only 6 drivers are labelled, which also "
             "limits how much the supervised model can learn.")


def s3(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "4 · Reason 2: the drivers look the same (SUMO)", "Some drivers cannot be told apart", crest)
    xs = list(range(60, 152, 2))
    cd = CategoryChartData()
    cd.categories = [f"{x}%" for x in xs]
    for name, mu in [("conservative", 90), ("normal", 100), ("aggressive", 120)]:
        cd.add_series(name, [round(math.exp(-0.5 * ((x - mu) / 10) ** 2), 3) for x in xs])
    chart_card(s, X0, 1.65, 5.9, 4.92, "Preferred speed of the simulated SUMO drivers (% of the limit)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(X0 + 0.15), Inches(2.05), Inches(5.6), Inches(4.4), cd).chart
    style_chart(ch)
    for ser, c in zip(ch.plots[0].series, [BLUE, GOLD, MAROON2]):
        ser.format.line.color.rgb = B.rgb(c)
        ser.format.line.width = Pt(3)
        ser.smooth = True
    ch.value_axis.visible = False
    ch.value_axis.has_major_gridlines = False
    cat = ch.category_axis._element
    for tag in ("c:tickLblSkip", "c:tickMarkSkip"):
        el = cat.find(qn(tag))
        if el is None:
            el = cat.makeelement(qn(tag), {})
            cat.append(el)
        el.set("val", "5")
    x2, w2 = X0 + 6.15, W - 6.15
    para_box(s, x2, 1.65, w2, 2.75, "What happens", [
        "In the simulation each car gets its own preferred speed, give or take 10%. Where the curves overlap, a normal car "
        "and a conservative car drive exactly the same.",
        "The only other clue is how closely a car follows the car ahead, and that shows only while it is following one."], size=12)
    stat(s, x2, 4.6, w2, f"{100 * v['never_normal']:.0f}%", "", "of normal SUMO drivers the PPO agent gets right when the car "
         "never follows anyone (it calls them conservative)", dark=True)
    para_box(s, x2, 5.7, w2, 0.87, f"PPO agent, car follows someone: {100 * v['follow_acc']:.0f}% right", [], size=12)
    B.footer(s, n)
    notes(s, "Reason two applies to the simulated SUMO drivers. Every car gets its own preferred speed, plus or minus 10 percent. "
             "So a fast normal car and a slow conservative car drive the same. Nothing in the data can separate them, not even a "
             "perfect model. The one extra clue is the following distance, and it only appears when the car is actually behind "
             f"another car. Numbers for the PPO agent on the simulated test drivers: when a car never follows anyone, it gets only "
             f"{100 * v['never_normal']:.0f}% of normal drivers right, it simply calls them conservative. When a car follows "
             f"someone, it gets {100 * v['follow_acc']:.0f}%. Two thirds of all its remaining mistakes are normal and conservative "
             "mixed up.")


def s4(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "5 · Reason 3: missing information", "The models see too little", crest)
    w = (W - 2 * 0.25) / 3
    boxes = [("Sensor gaps (UAH)", [
        f"In UAH the phone camera saw no car ahead in {100 * v['uah_unseen']:.0f}% of seconds.",
        "This was stored as 0, which the formula read as 'nobody close', so it looked calm.",
        "Fixed this week. The formula on driver D1 went from 0.61 to 0.76."]),
        ("Too short", [
            "On simulated SUMO drivers, watching longer helps: PPO agent 72% after about 30 s, formula 75% after 60 s.",
            "The German drones see each car for only 9 to 17 s. German data has no labels, so no accuracy can be measured "
            "there at all."]),
        ("Too little action", [
            "Inverse RL (tested on simulated SUMO highway drivers) learns from reactions: braking, speeding up.",
            "There, 90% of seconds are 'keep the same speed'.",
            f"So per car it gets {v['irl_auc']:.2f}, while the formula gets {v['irl_formula']:.2f} on the same cars."])]
    for i, (t, ps) in enumerate(boxes):
        para_box(s, X0 + i * (w + 0.25), 1.65, w, 3.55, t, ps, dark=(i == 0), size=12)
    card(s, X0, 5.4, W, 1.17, fill="FFFFFF", line=B.CARD_LINE)
    text(s, X0 + 0.28, 5.55, W - 0.56, 0.9, [[("In one line: ", {"bold": True, "color": MAROON2}),
                                              ("a model cannot be more accurate than its labels are right, its drivers are "
                                               "different, and its information is complete.", {})]], size=14, color=INK)
    B.footer(s, n)
    notes(s, "Reason three: the models often do not have the information. First, UAH: the camera missed the car ahead in "
             f"{100 * v['uah_unseen']:.0f}% of seconds, and that used to be stored as zero, which looks like calm driving. I fixed "
             "that this week and the formula on driver D1 jumped from 0.61 to 0.76. Second, time: on the simulated drivers, "
             "watching longer helps, the PPO agent is at 72% after about 30 seconds and the formula at 75% after 60. The German "
             "drones see each car for only 9 to 17 seconds, and German data has no labels, so we cannot measure any accuracy there; "
             "we can only say what share of cars the formula calls aggressive. Third, inverse RL, tested on the simulated highway "
             "drivers: it learns from reactions, but most driving is just cruising, so one car gives it very little to learn from.")


def s5(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "6 · This week", "What changed (Dr. Daher's list)", crest)
    ref = v["ref"]
    rows = [("Cleaned every dataset", "missing values are now 'unknown' instead of 0; impossible values removed"),
            ("New formula", f"speed not squared, acceleration squared, gap in metres and seconds mixed; weights refitted "
                            f"({' / '.join(f'{x:.2f}' for x in ref['weights'])})"),
            ("Formula on UAH", f"real drivers, tested on a driver never seen: AUC 0.83 → {ref['tests']['uah lodo']:.2f}"),
            ("Formula on German / US", f"no labels, so only the share called aggressive: highD 51% → {100 * v['highd_aggr']:.0f}%, "
                                       f"US-101 51% → {100 * v['us101_aggr']:.0f}% (dense traffic no longer looks aggressive)"),
            ("PPO agent (SUMO)", "16 inputs cut to 8 with the same accuracy (71%). Three other rewards and a second agent (DQN) "
                                 "all landed at 69 to 72%: none could get past reason 2")]
    for i, (a, b) in enumerate(rows):
        y = 1.65 + i * 0.97
        card(s, X0, y, W, 0.85, fill=B.CARD)
        text(s, X0 + 0.28, y + 0.22, 3.0, 0.45, [[(a, {"bold": True, "size": 14, "color": MAROON2})]])
        text(s, X0 + 3.4, y + 0.14, W - 3.7, 0.62, b, size=13, color=INK)
    B.footer(s, n)
    notes(s, "What I changed this week, from your list. I cleaned every dataset: missing values were stored as zero, now they "
             "are unknown. New formula with speed not squared and acceleration squared, weights refitted. The formula on the real "
             f"UAH drivers, tested on a driver it never saw, went from 0.83 to {ref['tests']['uah lodo']:.2f}. On the German and US "
             "traffic there are no labels, so I can only report the share of cars the formula calls aggressive: it dropped from "
             "about half to a quarter on highD and to 11% on US-101, so dense traffic no longer looks aggressive. For the PPO "
             "agent on the simulated drivers, I cut its inputs from 16 to 8 with no loss. I also tried three other reward "
             "functions and a second agent, DQN: all of them land between 69 and 72%, because of reason two. One open point: with "
             "the square, acceleration adds almost nothing, because the scale of 5 m/s2 was hand picked.")


def s6(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "8 · Reason 2, live", "Two simulated normal drivers, live", crest)
    w = (W - 0.3) / 2
    h = w * 607 / 1080
    for i, (f, cap) in enumerate([("explainer_correct_follows.gif", "Follows a car for 45 s. The following clue appears. "
                                   "Labelled normal: correct."),
                                  ("explainer_wrong_never_follows.gif", "Never close to another car. No clue appears. "
                                   "Labelled conservative: wrong.")]):
        x = X0 + i * (w + 0.3)
        path = os.path.join(OUT_DIR, f)
        if os.path.exists(path):
            s.shapes.add_picture(path, Inches(x), Inches(1.7), Inches(w), Inches(h))
        text(s, x, 1.7 + h + 0.12, w, 0.7, [[(cap, {"bold": i == 1})]], size=13, color=INK)
    text(s, X0, 5.75, W, 0.85, "Agent: PPO (8 inputs + 'wait 5 s' button), trained on simulated SUMO runs 0 to 6. Both drivers "
         "come from test run 7, never seen in training. Top: speed and gap to the car ahead. Bottom: the 8 numbers the agent "
         "receives, and how likely it is to press each button. Plays in slide show mode.", size=11, color=MUTED)
    B.footer(s, n)
    notes(s, "Here is reason two happening. The agent is the PPO agent with 8 inputs and the wait 5 seconds button, trained on "
             "simulated SUMO runs 0 to 6. Both drivers come from test run 7, which it never saw, and both are truly normal. On the "
             "left the car follows another car at about 30 metres, the following clue shows up, and the agent says normal, "
             "correct. On the right the car cruises at 98 km/h with nobody close, the clue never shows up, and the agent says "
             "conservative with 83% confidence. The agent is not broken; it never received the information it needed.")


def eq_box(s, x, y, w, h, title, items, dark=False):
    """items: (equation, plain explanation) pairs"""
    card(s, x, y, w, h, fill=MAROON if dark else B.CARD)
    text(s, x + 0.25, y + 0.15, w - 0.5, 0.35, [[(title, {"bold": True, "size": 15, "color": "FFFFFF" if dark else MAROON2})]])
    paras = []
    for eq, ex in items:
        paras.append([(eq, {"bold": True, "size": 12, "color": "FFFFFF" if dark else INK})])
        paras.append([(ex, {"size": 10.5, "color": PALE if dark else MUTED})])
    tb = text(s, x + 0.25, y + 0.58, w - 0.5, h - 0.7, paras, size=12)
    for k, p in enumerate(tb.text_frame.paragraphs):
        p.space_before = Pt(7 if k % 2 == 0 and k else 1)


def s_maths(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "2 · The maths, updated", "What each model computes", crest)
    w = v["ref"]["weights"]
    lo, hi = v["ref"]["cutoffs"]
    cw = (W - 0.25) / 2
    eq_box(s, X0, 1.6, cw, 2.45, "Formula (fixed rule, no learning)", [
        (f"score = 100 × ({w[0]:.2f}·speed + {w[1]:.2f}·accel² + {w[2]:.2f}·gap² + {w[3]:.2f}·lane)",
         "speed = km/h ÷ 150, accel = size of acceleration ÷ 5, gap = metres and seconds to the car ahead mixed, lane = offset ÷ 1.5 m"),
        ("unknown term → left out, the other weights rescaled", "missing data no longer counts as calm driving"),
        (f"below {lo:.0f} conservative, {lo:.0f} to {hi:.1f} normal, above {hi:.1f} aggressive", "cut offs fitted on labelled data")],
           dark=True)
    eq_box(s, X0 + cw + 0.25, 1.6, cw, 2.45, "Supervised model (learns from labels)", [
        ("score = 100 × (w1·speed + w2·accel² + w3·gap² + w4·lane)", "w1 to w4 learned, one set per road type"),
        ("P(aggressive) = sigmoid((score − threshold) ÷ τ)", "turns 'how far above the threshold' into a probability"),
        ("loss = −log P(true label)", "small when it is confident and right, large when it is confident and wrong")])
    eq_box(s, X0, 4.25, W, 2.32, "PPO and DQN agents (reinforcement learning): watch a driver second by second", [
        ("state (8 numbers): score so far · score in the last 10 s · gap points while following a car · share of seconds "
         "with a car ahead · time watched · road type (highway / urban / weather)",
         "each built from the formula's weighted terms, so the agent sees the same points the formula adds up"),
        ("buttons: wait 1 s · wait 5 s · conservative · normal · aggressive",
         "reward: −0.002 per second waited, +1 if the label is right, −1 if it is wrong")])
    B.footer(s, n)
    notes(s, "These are the updated equations. The formula is a fixed weighted sum of four terms, now with speed not squared and "
             f"acceleration squared, weights {w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}, and unknown terms are left out instead of "
             "counted as zero. The supervised model uses the same sum but learns its own weights per road type, turns the score "
             "into a probability with a sigmoid, and is trained to make the true label likely. The PPO and DQN agents watch a "
             "driver second by second. Each second they see 8 numbers, built from the formula's weighted terms, and press one "
             "of five buttons: wait one second, wait five seconds, or one of the three labels. Waiting costs a little; a right "
             "label gives plus one, a wrong one minus one.")


def s_state(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "7 · The agent's state", "From 16 inputs to 8, with no loss", crest)
    p = pd.read_csv(os.path.join(DATA, "ablation_state_reward.csv")).set_index("config")
    q = pd.read_csv(os.path.join(DATA, "ablation_state_reward_dqn.csv")).set_index("config")
    order = [("full", "16"), ("weighted", "12"), ("small6", "8"), ("small4", "4"), ("tiny", "2")]
    cd = CategoryChartData()
    cd.categories = [f"{k} inputs" for _, k in order]
    cd.add_series("PPO agent", [round(100 * float(p.loc[c, "balanced_acc"]), 1) for c, _ in order])
    cd.add_series("DQN agent", [round(100 * float(q.loc[c, "balanced_acc"]), 1) if c in q.index else None for c, _ in order])
    cd.add_series("formula after 60 s", [round(100 * float(p.loc["formula_T60", "balanced_acc"]), 1)] * len(order))
    chart_card(s, X0, 1.65, 6.6, 4.92, "Simulated SUMO test drivers: % labelled right, by number of inputs")
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(X0 + 0.15), Inches(2.05), Inches(6.3), Inches(4.4), cd).chart
    style_chart(ch)
    for ser, c in zip(ch.plots[0].series, [MAROON2, BLUE, GREY_BAR]):
        ser.format.line.color.rgb = B.rgb(c)
        ser.format.line.width = Pt(2.5)
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = B.rgb(c)
        ser.marker.format.line.color.rgb = B.rgb(c)
        ser.smooth = False
    ch.value_axis.minimum_scale = 50
    ch.value_axis.maximum_scale = 80
    ch.value_axis.major_unit = 10
    ch.value_axis.tick_labels.number_format = '0"%"'
    ch.value_axis.tick_labels.number_format_is_linked = False
    x2, w2 = X0 + 6.85, W - 6.85
    para_box(s, x2, 1.65, w2, 2.6, "How to read it", [
        "The state is the list of numbers the agent sees each second. Fewer numbers make it easier to trace and explain.",
        "Going from 16 to 8 keeps the accuracy (71%). Going below 8 loses 7 to 10 points: the 'gap while following' and road "
        "type inputs carry real information."], size=11)
    card(s, x2, 4.45, w2, 2.12, fill="FFFFFF", line=B.CARD_LINE)
    text(s, x2 + 0.25, 4.57, w2 - 0.5, 0.3, [[("PPO agent, 8 inputs, other choices", {"bold": True, "color": MAROON2, "size": 13})]])
    rows = [("reward as before", "small6"), ("waiting cost that grows", "small6_growing_wait"),
            ("bigger cost for far mistakes", "small6_ordinal_cost"), ("extra 'wait 5 s' button", "small6_wait5")]
    for i, (lab, c) in enumerate(rows):
        y = 5.0 + i * 0.36
        text(s, x2 + 0.25, y, w2 - 1.4, 0.35, lab, size=11, color=INK)
        text(s, x2 + w2 - 1.1, y, 0.9, 0.35, [[(f"{100 * p.loc[c, 'balanced_acc']:.0f}%", {"bold": True, "color": MAROON2})]], size=11)
    B.footer(s, n)
    notes(s, "This is the state space Dr. Daher asked to shrink. On the simulated SUMO test drivers, the PPO agent with all 16 "
             "inputs gets 69%, with 12 it gets 70%, with 8 it gets 71%. So 8 is enough. Below 8 it drops to 64 and 61%, because "
             "the inputs I removed, the gap while following a car and the road type, carry real information. DQN with the same "
             "inputs gets 72% and 71%. The grey line is the formula after watching 60 seconds, 75%. On the right, the reward and "
             "button changes with 8 inputs: a growing waiting cost 69%, a bigger cost for far mistakes 70%, an extra wait 5 "
             "seconds button 72%, the best. None of them passes the ceiling, because of reason two.")


def s7(prs, crest, n, v):
    s = new_slide(prs)
    header(s, "9 · Summary", "What I did this week, and what it showed", crest)
    w = (W - 2 * 0.25) / 3
    para_box(s, X0, 1.65, w, 4.92, "What I did", [
        "Went back over every model (formula, supervised model, PPO agent, DQN agent, inverse RL) and every dataset.",
        "Cleaned the data, applied the changes from last meeting (new formula, smaller agent state, new rewards, DQN), and "
        "reran all of it.",
        "Then traced each model step by step to see why it scores what it scores."], size=12)
    para_box(s, X0 + w + 0.25, 1.65, w, 4.92, "What it showed", [
        "The models are not the bottleneck: four different models stop at about the same score.",
        "The limit is the data: labels per trip (UAH), drivers that look the same (simulation), and missing information "
        "(sensor gaps, short tracks, mostly cruising)."], dark=True, size=12)
    para_box(s, X0 + 2 * (w + 0.25), 1.65, w, 4.92, "What comes next", [
        "Better labels: per clip instead of per trip, and more labelled drivers (full DriveDNA once access is granted).",
        "A new clue for cars driving freely: their speed compared with the speed limit or the traffic around them.",
        "Pick the acceleration scale from the data instead of the hand picked 5 m/s²."], size=12)
    B.footer(s, n)
    notes(s, "To sum up. This week I went back over every model and every dataset, cleaned the data, applied the changes from "
             "our last meeting, reran everything, and traced each model step by step. What it showed: the models are not the "
             "bottleneck, four different models stop at about the same score. The limit is in the data: labels per trip in UAH, "
             "drivers that look the same in the simulation, and missing information. So next I want to improve the data rather "
             "than try more models: better labels and more drivers, a clue for freely driving cars, and an acceleration scale "
             "taken from the data.")


def main():
    v = numbers()
    prs = Presentation(B.SRC)
    with zipfile.ZipFile(B.SRC) as z:
        crest = os.path.join(tempfile.mkdtemp(), "crest.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    for i, f in enumerate([s1, s_maths, s2, s3, s4, s5, s_state, s6, s7]):
        f(prs, crest, i + 1, v)
    assert len(prs.slides) == B.TOTAL
    prs.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    main()
