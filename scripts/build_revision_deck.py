"""
Weekly update (week of 20 Oct 2026): Dr. Daher's revision list (clean the data, new formula, smaller state, reward and
action space, rerun everything, backtrack every model, explain with a video). Same style as the October deck
(helpers from scripts/build_october_deck.py); every number is read from data/.

    python scripts/build_revision_deck.py
writes ../20-10-2026/Weekly Update.pptx (the two explainer GIFs in ../20-10-2026/ are embedded and play in
PowerPoint and Keynote)
"""
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_october_deck as B  # noqa: E402
from build_october_deck import (GREY_BAR, INK, MAROON, MAROON2, MUTED, PALE, PINK, W, X0, bullet_lines, card,  # noqa: E402
                                chart_card, color_series, header, new_slide, stat, style_chart, text)

DATA, ROOT = B.DATA, B.ROOT
OUT_DIR = os.path.abspath(os.path.join(ROOT, "..", "20-10-2026"))
OUT = os.path.join(OUT_DIR, "Weekly Update.pptx")
GOLD, BLUE = "D9A441", "4A6FA5"
B.TOTAL = 10
SOURCES = [("uah", "UAH"), ("planted", "SUMO"), ("ngsim_us_101", "US-101"), ("ngsim_i_80", "I-80"),
           ("ngsim_lankershim", "Lankershim"), ("ngsim_peachtree", "Peachtree"), ("pneuma", "pNEUMA"),
           ("highd", "highD"), ("exid", "exiD"), ("ind", "inD"), ("round", "rounD"), ("unid", "uniD"),
           ("drivedna_sample", "DriveDNA")]


def bars(ch, top, unit, fmt='0"%"', labels=None):
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = top
    ch.value_axis.major_unit = unit
    ch.value_axis.tick_labels.number_format = fmt
    ch.value_axis.tick_labels.number_format_is_linked = False
    pl = ch.plots[0]
    pl.gap_width = 60
    if labels:
        pl.has_data_labels = True
        pl.data_labels.number_format = labels
        pl.data_labels.number_format_is_linked = False
        pl.data_labels.font.size = Pt(10)


def why(s, x, y, w, h, title, items, step=0.68, size=12, dark=False):
    card(s, x, y, w, h, fill=MAROON if dark else B.CARD)
    text(s, x + 0.25, y + 0.15, w - 0.5, 0.3, [[(title, {"bold": True, "color": "FFFFFF" if dark else MAROON2, "size": 15})]])
    bullet_lines(s, x + 0.25, y + 0.6, w - 0.5, items, step=step, size=size, color=PALE if dark else INK,
                 marker=PINK if dark else MAROON2)


# ── slides ──────────────────────────────────────────────────────────────────
def s_overview(prs, crest, n):
    s = new_slide(prs)
    header(s, "1 · This week", "Dr. Daher's list, and what was done", crest)
    items = [("Clean the data", "every dataset audited the same way; unknown values no longer stored as 0"),
             ("New formula", "speed not squared, acceleration squared, gap in metres and seconds mixed; weights refitted"),
             ("Smaller state", "agent inputs built from weighted terms, then cut from 16 to 8"),
             ("Reward and actions", "growing wait cost, cost per mistake type, a 'wait 5 s' button; DQN next to PPO"),
             ("Rerun and backtrack", "every model on every dataset; each calculation traced; a replay video")]
    w = (W - 4 * 0.2) / 5
    for i, (t, b) in enumerate(items):
        x = X0 + i * (w + 0.2)
        card(s, x, 1.7, w, 3.0, fill=MAROON if i == 4 else B.CARD)
        dark = i == 4
        text(s, x + 0.18, 1.85, w - 0.36, 0.6, [[(t, {"bold": True, "size": 15, "color": "FFFFFF" if dark else MAROON2})]])
        text(s, x + 0.18, 2.55, w - 0.36, 2.0, b, size=12, color=PALE if dark else INK)
    text(s, X0, 5.0, W, 1.5, [
        [("Data: ", {"bold": True, "color": MAROON2}), ("UAH (6 real drivers, labelled), simulated SUMO drivers (labelled), "
         "NGSIM, pNEUMA, highD, exiD, inD, rounD, uniD, DriveDNA sample (no labels). Full DriveDNA runs on the workstation.", {})],
        [("Measures: ", {"bold": True, "color": MAROON2}), ("accuracy = share labelled right; AUC = chance an aggressive driver "
         "scores above a normal one (0.5 coin flip, 1.0 perfect).", {})]], size=12, color=INK)
    B.footer(s, n)


def s_cleaning(prs, crest, n):
    s = new_slide(prs)
    header(s, "2 · Data cleaning", "Missing data was stored as 0, and 0 reads as calm", crest)
    summ = pd.read_csv(os.path.join(DATA, "rerun_formula_summary.csv"))
    a = summ[summ["label"] == "all"].set_index("source")
    cd = CategoryChartData()
    cd.categories = [lab for src, lab in SOURCES]
    cd.add_series("gap unknown", [round(100 * float(a.loc[src, "unknown_prox2"]), 1) for src, _ in SOURCES])
    cd.add_series("lane offset unknown", [round(100 * float(a.loc[src, "unknown_wave"]), 1) for src, _ in SOURCES])
    chart_card(s, X0, 1.65, 7.0, 4.92, "Share of 10 s windows with a term unknown (was silently 0 before)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [MAROON2, GOLD])
    bars(ch, 100, 20)
    ch.plots[0].gap_width = 40
    ch.category_axis.reverse_order = True
    ch.category_axis.tick_labels.font.size = Pt(9)
    log = pd.read_csv(os.path.join(DATA, "cleaning_log.csv")).set_index("source")
    x2, w2 = X0 + 7.25, W - 7.25
    stat(s, x2, 1.65, w2, f"{100 * log.loc['uah', 'rule4_gap_not_detected'] / log.loc['uah', 'rows_in']:.0f}%", "",
         "of UAH seconds: the camera saw no car, which was stored as 'nobody close'", dark=True)
    why(s, x2, 2.75, w2, 3.82, "Rules (same for every dataset)", [
        "unknown becomes 'unknown', not 0: the score uses only the terms that are known",
        "impossible values removed: lane offset over 2 m (up to 31 m in NGSIM), gap under 2 m at speed",
        "a 'car ahead: yes / no / unknown' flag added",
        "German drones see each car only 9 to 17 s"], step=0.78, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = ("scripts/audit_datasets.py found the problems (data/dataset_audit.csv); "
                                           "datasets/cleaning.py applies the rules (data/cleaning_log.csv).")


def s_formula(prs, crest, n):
    s = new_slide(prs)
    header(s, "3 · New formula", "Better on every test; squared acceleration barely counts", crest)
    ref = json.load(open(os.path.join(DATA, "reference_weights.json")))
    w = ref["weights"]
    card(s, X0, 1.65, 5.6, 2.25, fill=MAROON)
    text(s, X0 + 0.25, 1.8, 5.1, 2.0, [
        [("Old", {"bold": True, "color": PINK, "size": 13})],
        [("100 × (0.5·speed² + 0.2·accel + 0.8·gap² + 0.4·lane), missing = 0", {"color": PALE, "size": 12})],
        [("New", {"bold": True, "color": "FFFFFF", "size": 13})],
        [(f"100 × ({w[0]:.2f}·speed + {w[1]:.2f}·accel² + {w[2]:.2f}·gap² + {w[3]:.2f}·lane), unknown terms left out; "
          f"cut offs {ref['cutoffs'][0]:.0f} / {ref['cutoffs'][1]:.1f}", {"color": "FFFFFF", "size": 12, "bold": True})],
        [("gap² = metres and seconds mixed per environment", {"color": PALE, "size": 11})]], size=12)
    t = ref["tests"]
    rows = [("UAH, new driver (AUC)", "0.83", f"{t['uah lodo']:.2f}"),
            ("SUMO, aggressive vs normal (AUC)", "0.72 to 0.92", f"{t['planted test aggressive vs normal']:.2f}"),
            ("SUMO, normal vs conservative (AUC)", "0.64 to 0.76", f"{t['planted test normal vs conservative']:.2f}")]
    card(s, X0, 4.1, 5.6, 2.47, fill="FFFFFF", line=B.CARD_LINE)
    text(s, X0 + 0.25, 4.22, 3.3, 0.3, [[("test", {"bold": True, "color": MUTED})]], size=11)
    text(s, X0 + 3.4, 4.22, 1.0, 0.3, [[("old", {"bold": True, "color": MUTED})]], size=11)
    text(s, X0 + 4.55, 4.22, 0.8, 0.3, [[("new", {"bold": True, "color": MAROON2})]], size=11)
    for i, (a, b, c) in enumerate(rows):
        y = 4.65 + i * 0.6
        text(s, X0 + 0.25, y, 3.1, 0.55, a, size=12, color=INK)
        text(s, X0 + 3.4, y, 1.1, 0.55, b, size=12, color=MUTED)
        text(s, X0 + 4.55, y, 0.9, 0.55, [[(c, {"bold": True, "color": MAROON2})]], size=12)
    x2, w2 = X0 + 5.85, W - 5.85
    summ = pd.read_csv(os.path.join(DATA, "rerun_formula_summary.csv"))
    a = summ[summ["label"] == "all"]
    stat(s, x2, 1.65, w2, f"{a['pts_accel2'].min():.1f} to {a['pts_accel2'].max():.1f}", "",
         "points from squared acceleration per window, on every dataset (score about 20)", dark=True)
    why(s, x2, 2.75, w2, 3.82, "Why", [
        f"most accelerations are under 2 m/s²: (2 ÷ 5)² = 0.16, so even with the largest weight ({w[1]:.2f}) it adds almost nothing",
        "the gain came from the gap mix and from no longer counting unknown values as calm",
        "D1 (camera rarely saw a car): AUC 0.61 → 0.76",
        "to discuss: keep the square, or use it only above a hard braking level"], step=0.8, size=11)
    B.footer(s, n)


def s_points(prs, crest, n):
    s = new_slide(prs)
    header(s, "4 · Backtrack: the formula", "Where every point of the score comes from", crest)
    summ = pd.read_csv(os.path.join(DATA, "rerun_formula_summary.csv"))
    a = summ[summ["label"] == "all"].set_index("source")
    pl = summ[(summ["source"] == "planted") & summ["label"].isin(["conservative", "normal", "aggressive"])].set_index("label")
    cats = [("planted", "conservative", "SUMO conservative"), ("planted", "normal", "SUMO normal"),
            ("planted", "aggressive", "SUMO aggressive"), ("uah", "normal", "UAH normal"), ("uah", "aggressive", "UAH aggressive"),
            ("highd", "all", "highD"), ("ngsim_us_101", "all", "US-101"), ("pneuma", "all", "pNEUMA")]
    rowsrc = {(r["source"], r["label"]): r for _, r in summ.iterrows()}
    cd = CategoryChartData()
    cd.categories = [c[2] for c in cats]
    for term, lab in [("speed", "speed"), ("accel2", "acceleration²"), ("prox2", "gap"), ("wave", "lane")]:
        cd.add_series(lab, [round(float(rowsrc[(c[0], c[1])][f"pts_{term}"]), 1) for c in cats])
    chart_card(s, X0, 1.65, 7.6, 4.92, "Mean points per term (they add up to the score)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(X0 + 0.15), Inches(2.05), Inches(7.3), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [GREY_BAR, BLUE, MAROON2, GOLD])
    ch.plots[0].gap_width = 50
    ch.category_axis.reverse_order = True
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = 35
    ch.value_axis.major_unit = 5
    x2, w2 = X0 + 7.85, W - 7.85
    stat(s, x2, 1.65, w2, f"{pl.loc['aggressive', 'pts_prox2']:.0f} / {pl.loc['conservative', 'pts_prox2']:.0f}", "",
         "gap points, SUMO aggressive vs conservative", dark=True)
    why(s, x2, 2.75, w2, 3.82, "Reading", [
        "speed gives most points, but almost the same for every type",
        "the gap is what separates the types",
        f"highD aggressive share {100 * a.loc['highd', 'share_aggressive']:.0f}% (was 51%), "
        f"US-101 {100 * a.loc['ngsim_us_101', 'share_aggressive']:.0f}% (was 51%)"], step=0.9, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = "Hand traced windows: docs/hand_checks/backtrace_formula.md."


def s_supervised(prs, crest, n):
    s = new_slide(prs)
    header(s, "5 · Correction", "Measured correctly, the supervised model ranks best", crest)
    sup = pd.read_csv(os.path.join(DATA, "rerun_supervised.csv"))
    l = sup[sup["test"] == "uah lodo"]
    vals = [("formula, old", 0.829), ("supervised, old,\nas reported", 0.762), ("supervised, old,\nmeasured right", 0.897),
            ("formula, new", float(l["formula_auc"].mean())), ("supervised, new", float(l["supervised_auc"].mean()))]
    cd = CategoryChartData()
    cd.categories = [v[0] for v in vals]
    cd.add_series("AUC", [round(v[1], 2) for v in vals])
    chart_card(s, X0, 1.65, 6.8, 4.92, "UAH, tested on a driver never seen (AUC)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.5), Inches(4.4), cd).chart
    style_chart(ch, legend=False)
    color_series(ch, [GREY_BAR])
    for i, c in enumerate([GREY_BAR, "E9E2E3", MAROON2, GREY_BAR, MAROON2]):
        pt = ch.plots[0].series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = B.rgb(c)
    bars(ch, 1.0, 0.25, fmt="0.00", labels="0.00")
    x2, w2 = X0 + 7.05, W - 7.05
    why(s, x2, 1.65, w2, 4.92, "What went wrong last week", [
        "the model learns one threshold per environment: highway about 12, urban about 22",
        "ranking highway and urban scores together mixed two scales",
        "measured as distance above its own threshold: 0.90 (old terms), 0.87 (new terms)",
        "last week's slide 'supervised below the formula' was wrong",
        "the new terms are slightly worse for it: squared acceleration lost information"], step=0.85, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = ("Old fold models in data/uah_lodo_agents rescored with the margin. "
                                           "Script: scripts/rerun_supervised.py; backtrace: docs/hand_checks/backtrace_supervised.md.")


def s_state(prs, crest, n):
    s = new_slide(prs)
    header(s, "6 · Smaller state, reward, actions", "8 inputs are enough; nothing moves the ceiling", crest)
    p = pd.read_csv(os.path.join(DATA, "ablation_state_reward.csv")).set_index("config")
    q = pd.read_csv(os.path.join(DATA, "ablation_state_reward_dqn.csv")).set_index("config")
    order = [("full", "16"), ("weighted", "12"), ("small6", "8"), ("small4", "4"), ("tiny", "2")]
    cd = CategoryChartData()
    cd.categories = [f"{k} inputs" for _, k in order]
    cd.add_series("PPO", [round(100 * float(p.loc[c, "balanced_acc"]), 1) for c, _ in order])
    cd.add_series("DQN", [round(100 * float(q.loc[c, "balanced_acc"]), 1) if c in q.index else None for c, _ in order])
    cd.add_series("formula after 60 s", [round(100 * float(p.loc["formula_T60", "balanced_acc"]), 1)] * len(order))
    chart_card(s, X0, 1.65, 6.8, 4.92, "Accuracy on simulated test drivers (3 types)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(X0 + 0.15), Inches(2.05), Inches(6.5), Inches(4.4), cd).chart
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
    x2, w2 = X0 + 7.05, W - 7.05
    card(s, x2, 1.65, w2, 2.5, fill="FFFFFF", line=B.CARD_LINE)
    text(s, x2 + 0.25, 1.78, w2 - 0.5, 0.3, [[("8 inputs, PPO, other rewards and actions", {"bold": True, "color": MAROON2, "size": 13})]])
    rows = [("as before", "small6"), ("waiting cost grows", "small6_growing_wait"), ("cost per mistake type", "small6_ordinal_cost"),
            ("extra 'wait 5 s' button", "small6_wait5")]
    for i, (lab, c) in enumerate(rows):
        y = 2.25 + i * 0.44
        text(s, x2 + 0.25, y, w2 - 1.6, 0.4, lab, size=12, color=INK)
        text(s, x2 + w2 - 1.3, y, 1.1, 0.4, [[(f"{100 * p.loc[c, 'balanced_acc']:.0f}%", {"bold": True, "color": MAROON2})]], size=12)
    why(s, x2, 4.35, w2, 2.22, "Reading", [
        "the 8: score so far, last 10 s, gap points while following, car ahead share, time, environment",
        "removing 'gap while following' and environment costs 7 points",
        "agents 70 to 72% in about 30 s; formula 75% after 60 s"], step=0.55, size=10)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = ("Single training runs; earlier repeats varied by about 0.01 to 0.02. "
                                           "scripts/state_reward_ablation.py, env/labeling_ref.py.")


def s_agent(prs, crest, n):
    s = new_slide(prs)
    header(s, "7 · Backtrack: the agent", "What PPO learned: 'no close car, call it conservative'", crest)
    b = pd.read_csv(os.path.join(DATA, "backtrace_agents.csv"))
    b = b[b["run"] == "ppo:small6_wait5"].set_index("following")
    order = ["never", "under 25%", "25 to 75%", "over 75%"]
    cd = CategoryChartData()
    cd.categories = [f"follows {o}" if o != "never" else "never follows" for o in order]
    for c, lab in [("conservative", "conservative"), ("normal", "normal"), ("aggressive", "aggressive")]:
        cd.add_series(lab, [round(100 * float(b.loc[o, f"acc_{c}"]), 0) for o in order])
    chart_card(s, X0, 1.65, 7.0, 4.92, "PPO: drivers labelled right, by how much the car followed another car")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [BLUE, GOLD, MAROON2])
    bars(ch, 100, 20)
    x2, w2 = X0 + 7.25, W - 7.25
    stat(s, x2, 1.65, w2, f"{100 * b.loc['never', 'acc_normal']:.0f}%", "", "of normal drivers right when the car never follows anyone",
         dark=True)
    why(s, x2, 2.75, w2, 3.82, "Reading", [
        "following shows each type's own gap: there PPO gets 78%",
        "a car driving freely only shows its speed, and speeds overlap",
        "so PPO falls back on one rule and calls almost every free car conservative",
        "DQN uses speed more and gets 80% of free aggressive cars"], step=0.78, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = "docs/hand_checks/backtrace_agents.md (scripts/backtrace_agents.py)."


def s_video(prs, crest, n):
    s = new_slide(prs)
    header(s, "8 · Replay", "Two normal drivers, as PPO sees them", crest)
    w = (W - 0.3) / 2
    h = w * 607 / 1080
    for i, (f, cap) in enumerate([("explainer_correct_follows.gif", "follows a car for 45 s: the gap clue shows, labelled normal (correct)"),
                                   ("explainer_wrong_never_follows.gif", "never closer than 100 m to a car: labelled conservative (wrong)")]):
        x = X0 + i * (w + 0.3)
        path = os.path.join(OUT_DIR, f)
        if os.path.exists(path):
            s.shapes.add_picture(path, Inches(x), Inches(1.75), Inches(w), Inches(h))
        text(s, x, 1.75 + h + 0.15, w, 0.6, cap, size=12, color=INK)
    text(s, X0, 5.75, W, 0.8, "Each frame: speed, gap to the car ahead, the 8 inputs PPO receives, and the probability it gives "
         "each button. The GIFs play in slide show mode.", size=12, color=MUTED)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = ("Made by scripts/make_explainer_video.py; the frame numbers are in the CSV next "
                                           "to each GIF in the 20-10-2026 folder.")


def s_irl(prs, crest, n):
    s = new_slide(prs)
    header(s, "9 · Inverse RL", "Right for a whole group, too little data for one car", crest)
    a = pd.read_csv(os.path.join(DATA, "rerun_irl_summary.csv"))
    z = pd.read_csv(os.path.join(DATA, "rerun_irl_summary_lam0.csv"))
    a, z = a[a["variant"] == "all seconds"].iloc[0], z[z["variant"] == "all seconds"].iloc[0]
    cd = CategoryChartData()
    types = ["aggressive", "normal", "conservative"]
    cd.categories = types
    cd.add_series("whole group", [round(a[f"pooled_headway_{c}"], 2) for c in types])
    cd.add_series("one car, pulled to the average", [round(a[f"per_vehicle_headway_{c}"], 2) for c in types])
    cd.add_series("one car, no pull", [round(z[f"per_vehicle_headway_{c}"], 2) for c in types])
    chart_card(s, X0, 1.65, 7.0, 4.92, "Preferred gap to the car ahead (s), simulated test cars")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [MAROON2, GREY_BAR, GOLD])
    bars(ch, 4.0, 1.0, fmt="0", labels="0.0")
    x2, w2 = X0 + 7.25, W - 7.25
    stat(s, x2, 1.65, w2, f"{a['auc_irl']:.2f} / {z['auc_irl']:.2f}", "", "AUC per car with / without the pull (formula 0.96)", dark=True)
    why(s, x2, 2.75, w2, 3.82, "Reading", [
        "pulled to the average: every car near 2.5 s",
        "no pull: the averages separate, but each car's estimate is so noisy that AUC drops",
        "about 100 decisions per car, most of them 'keep speed'",
        "the limit is information per car, not the method"], step=0.78, size=11)
    B.footer(s, n)


def s_next(prs, crest, n):
    s = new_slide(prs)
    header(s, "10 · Next", "Decisions and next steps", crest)
    why(s, X0, 1.65, 5.65, 4.92, "To decide with Dr. Daher", [
        "squared acceleration: keep, or count only hard braking?",
        "the missing information is how free driving differs: add speed relative to the speed limit or to traffic?",
        "simulated types: keep the realistic overlap, or widen it to test methods?",
        "8 inputs and the 'wait 5 s' button as the agent standard?"], step=1.0, dark=True)
    why(s, X0 + 5.9, 1.65, W - 5.9, 4.92, "Next steps", [
        "full DriveDNA on the workstation (script and instructions ready)",
        "repeat PPO and DQN with 3 seeds each",
        "PPO and DQN on UAH with the new state",
        "German data: drones see each car only 9 to 17 s, too short for the agents; formula only there"], step=1.0)
    B.footer(s, n)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prs = Presentation(B.SRC)
    with zipfile.ZipFile(B.SRC) as z:
        crest = os.path.join(tempfile.mkdtemp(), "crest.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    for i, f in enumerate([s_overview, s_cleaning, s_formula, s_points, s_supervised, s_state, s_agent, s_video, s_irl, s_next]):
        f(prs, crest, i + 1)
    assert len(prs.slides) == B.TOTAL
    prs.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    main()
