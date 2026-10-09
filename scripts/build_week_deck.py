"""
Short deck for the next meeting with Dr. Daher (week of 7 Oct 2026), same style as the October deck
(helpers from scripts/build_october_deck.py). Every number is read from data/ or docs/hand_checks/.

    python scripts/build_week_deck.py
writes ../28:9:2026/Presentation/VIPP 301A Update 7 October 2026.pptx
"""
import json
import os
import re
import sys
import tempfile
import zipfile

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_october_deck as B  # noqa: E402
from build_october_deck import (CARD_LINE, GREY_BAR, HEAD, INK, MAROON, MAROON2, MUTED, PALE, PINK, W, WINE, X0,  # noqa: E402
                                bullet_lines, card, chart_card, color_series, header, new_slide, rgb, stat,
                                style_chart, text)

DATA, ROOT = B.DATA, B.ROOT
OUT = os.path.join(B.PRES, "VIPP 301A Update 7 October 2026.pptx")
GOLD = "D9A441"
B.TOTAL = 8


def footer(s, n):
    B.footer(s, n)


def summary(tag):
    return pd.read_csv(os.path.join(DATA, f"labeling_rl_planted_summary{tag}.csv"), index_col=0)


def title_slide(prs, logo):
    s = new_slide(prs, MAROON)
    s.shapes.add_picture(logo, Inches(X0), Inches(0.667), Inches(3.333), Inches(0.778))
    text(s, 8.2, 0.6, 4.244, 1.0, [[("Hadi Al Shmaissani", {"bold": True, "color": "FFFFFF"})],
                                    [("Supervisor: Dr. Naseem Daher", {"color": PALE})],
                                    [("October 2026 · American University of Beirut", {"color": PALE})]],
         size=12, align=PP_ALIGN.RIGHT)
    text(s, X0, 1.85, W, 0.3, "VIPP 301A · UPDATE, WEEK OF 7 OCTOBER", size=12, color=PINK, spacing=150)
    text(s, X0, 2.2, W, 0.8, [[("Proximity by environment, and RL that decides when to label",
                                {"font": HEAD, "size": 36, "color": "FFFFFF"})]])
    steps = [("1 · Correction", "what the index model really is"),
             ("2 · Proximity", "metres and headway mixed per environment"),
             ("3 · RL labeler", "watch a driver, decide when to label"),
             ("4 · Simulation", "US-101 congestion with a lane drop")]
    w = (W - 3 * 0.3) / 4
    for i, (a, b) in enumerate(steps):
        x = X0 + i * (w + 0.3)
        card(s, x, 4.45, w, 1.35, fill=WINE)
        text(s, x + 0.2, 4.62, w - 0.4, 0.3, [[(a, {"bold": True, "color": "FFFFFF", "size": 14})]])
        text(s, x + 0.2, 5.0, w - 0.4, 0.7, b, size=12, color=PALE)
    return s


def slide_correction(prs, crest, n):
    s = new_slide(prs)
    header(s, "1 · Correction", "The index model is supervised; the RL is new", crest)
    cols = [("Aggressiveness Index", "closed form formula",
             ["4 normalised terms: speed, acceleration, proximity, lane offset",
              "fixed weights 0.5 / 0.2 / 0.8 / 0.4, score 0 to 100",
              "cut offs 29 / 42 fitted on labelled data"]),
            ("DynamicWeightAgent", "supervised, about 15 parameters",
             ["one weight set and one threshold per environment",
              "logistic loss on UAH normal / aggressive trips",
              "spring reports called it an RL-trained MLP: the code is not"]),
            ("Labeling agent (new)", "reinforcement learning, PPO",
             ["does not drive: it watches a driver second by second",
              "each second: wait (small cost) or commit to a label",
              "old 'Q-learner' was label counting, renamed score_bucket_labeler"])]
    w = (W - 2 * 0.3) / 3
    for i, (t, sub, items) in enumerate(cols):
        x = X0 + i * (w + 0.3)
        dark = i == 2
        card(s, x, 1.75, w, 4.75, fill=MAROON if dark else B.CARD)
        text(s, x + 0.3, 1.95, w - 0.6, 0.35, [[(t, {"bold": True, "size": 16, "color": "FFFFFF" if dark else MAROON2})]])
        text(s, x + 0.3, 2.35, w - 0.6, 0.3, sub, size=12, color=PINK if dark else MUTED)
        bullet_lines(s, x + 0.3, 2.95, w - 0.6, items, color=PALE if dark else INK, marker=PINK if dark else MAROON2, step=1.0)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = ("Code audit, 5 Oct: nothing on the index side was RL. The driving agent in SUMO is "
                                           "RL but belongs to the driving part of the project.")
    return s


def slide_mix(prs, crest, n):
    s = new_slide(prs)
    header(s, "2 · Proximity", "A per-environment mix beats both measures", crest)
    g = pd.read_csv(os.path.join(DATA, "proximity_mix_grid.csv"))
    alpha = json.load(open(os.path.join(DATA, "proximity_mix.json")))["alpha_metres"]
    cd = CategoryChartData()
    cd.categories = [f"{a:.1f}" for a in sorted(g["alpha"].unique())]
    for env, col in [("highway", "auc_planted_fit"), ("urban", "auc_planted_fit"), ("weather", "auc_planted_fit")]:
        cd.add_series(f"{env}, SUMO", [round(float(v), 3) for v in g[g["environment"] == env].sort_values("alpha")[col]])
    cd.add_series("highway, UAH", [round(float(v), 3) for v in g[g["environment"] == "highway"].sort_values("alpha")["auc_uah"]])
    chart_card(s, X0, 1.65, 7.0, 4.92, "AUC aggressive vs normal against share of metres (0 = headway, 1 = metres)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    ch.value_axis.minimum_scale = 0.7
    ch.value_axis.maximum_scale = 0.95
    ch.value_axis.major_unit = 0.05
    ch.value_axis.tick_labels.number_format = "0.00"
    ch.value_axis.tick_labels.number_format_is_linked = False
    for ser, c in zip(ch.plots[0].series, [MAROON2, GOLD, GREY_BAR, "4A6FA5"]):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(2.25)
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(c)
        ser.smooth = False
    hw = g[g["environment"] == "highway"].set_index("alpha")
    x2, w2 = X0 + 7.25, W - 7.25
    stat(s, x2, 1.65, w2, f"{hw.loc[alpha['highway'], 'auc_planted_fit']:.3f}", "",
         f"highway AUC with the mix, against {hw.loc[1.0, 'auc_planted_fit']:.3f} metres and {hw.loc[0.0, 'auc_planted_fit']:.3f} headway", dark=True)
    stat(s, x2, 2.75, w2, f"+{hw.loc[alpha['highway'], 'density_rise_normal']:.1f}", "",
         f"normal drivers, light traffic to jam (metres +{hw.loc[1.0, 'density_rise_normal']:.1f}, headway {hw.loc[0.0, 'density_rise_normal']:.1f})")
    card(s, x2, 3.85, w2, 2.72)
    text(s, x2 + 0.25, 4.0, w2 - 0.5, 0.3, [[("Fitted share of metres", {"bold": True, "color": MAROON2, "size": 14})]])
    bullet_lines(s, x2 + 0.25, 4.45, w2 - 0.5, [f"highway {alpha['highway']:.1f}", f"urban {alpha['urban']:.1f}",
                                                 f"weather {alpha['weather']:.1f}", "UAH alone still prefers metres"], step=0.48)
    footer(s, n)
    s.notes_slide.notes_text_frame.text = ("Dr. Daher: mix metres and headway depending on the environment. The share is "
                                           "fitted on planted SUMO drivers (seeds 0 to 6) and UAH; seeds 7 to 9 agree within 0.005.")
    return s


def slide_mdp(prs, crest, n):
    s = new_slide(prs)
    header(s, "3 · RL labeler", "Aggressiveness as a decision: when to commit", crest)
    boxes = [("Observe 1 s more", "running mean of the 4 terms, last 10 s, AI, time, environment"),
             ("Policy (PPO)", "wait, or label conservative / normal / aggressive"),
             ("Reward", "+1 correct, -1 wrong, -0.002 per second waited, 60 s horizon")]
    w = 3.3
    for i, (t, b) in enumerate(boxes):
        x = X0 + i * (w + 0.83)
        card(s, x, 2.0, w, 1.9, fill=MAROON if i == 1 else B.CARD)
        text(s, x + 0.25, 2.2, w - 0.5, 0.35, [[(t, {"bold": True, "size": 16, "color": "FFFFFF" if i == 1 else MAROON2})]])
        text(s, x + 0.25, 2.7, w - 0.5, 1.1, b, size=12, color=PALE if i == 1 else INK)
        if i < 2:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + w + 0.08), Inches(2.95), Inches(x + w + 0.75), Inches(2.95))
            c.line.color.rgb = rgb(MAROON2)
            c.line.width = Pt(2.5)
            c.line._get_or_add_ln().append(_arrow())
    text(s, X0 + 4.13, 4.0, 3.3, 0.3, "wait: back to observe", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    bullet_lines(s, X0, 4.6, W, [
        "Real RL: the action decides what is seen next and the reward comes later; supervised learning cannot learn when to stop.",
        "Two training details decide whether it works: no discount (0.99 per second makes a label after 60 s worth 0.55), "
        "and an initial wait bias (otherwise most decisions come after one second).",
        "Episodes: planted SUMO drivers with true types; test drivers from seeds 7 to 9, never seen in training."], step=0.62)
    footer(s, n)
    return s


def _arrow():
    from lxml import etree
    return etree.fromstring('<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle"/>')


def slide_rl_results(prs, crest, n):
    s = new_slide(prs)
    header(s, "3 · RL labeler", "Best fixed-window accuracy in half the time", crest)
    hw = [summary(f"_headway_wb4_seed{k}") for k in (0, 1, 2)]
    base = summary("_headway_wb4_seed0")
    ppo_acc = sum(float(d.loc["rl_ppo", "balanced_acc"]) for d in hw) / 3
    ppo_sec = sum(float(d.loc["rl_ppo", "seconds"]) for d in hw) / 3
    ppo_ag = sum(float(d.loc["rl_ppo", "acc_aggressive"]) for d in hw) / 3
    rows = [("index, 10 s", base.loc["refit_env_T10"]), ("index, 30 s", base.loc["refit_env_T30"]),
            ("index, 60 s", base.loc["refit_env_T60"])]
    cd = CategoryChartData()
    cd.categories = [r[0] for r in rows] + [f"PPO, {ppo_sec:.0f} s"]
    cd.add_series("balanced accuracy", [round(float(r[1]["balanced_acc"]), 3) for r in rows] + [round(ppo_acc, 3)])
    cd.add_series("aggressive drivers caught", [round(float(r[1]["acc_aggressive"]), 3) for r in rows] + [round(ppo_ag, 3)])
    chart_card(s, X0, 1.65, 6.6, 4.92, "Planted SUMO test drivers, headway features (PPO: mean of 3 seeds)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.3), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [MAROON2, GOLD])
    ch.value_axis.minimum_scale = 0.5
    ch.value_axis.maximum_scale = 0.95
    ch.value_axis.major_unit = 0.1
    ch.value_axis.tick_labels.number_format = "0.00"
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.plots[0].gap_width = 70
    ep = {t: pd.read_csv(os.path.join(DATA, f"labeling_rl_planted_episodes{t}.csv.gz")) for t in ("_headway_seed0", "_headway_wb4_seed0")}
    at1 = {t: float((d[d["method"] == "rl_ppo"]["seconds"] == 1).mean()) for t, d in ep.items()}
    x2, w2 = X0 + 6.85, W - 6.85
    stat(s, x2, 1.65, w2, f"{ppo_acc:.3f}", "", f"PPO balanced accuracy after {ppo_sec:.0f} s; fixed index "
         f"{float(base.loc['refit_env_T30', 'balanced_acc']):.3f} at 30 s", dark=True)
    stat(s, x2, 2.75, w2, f"{100 * at1['_headway_seed0']:.0f}%", f"→{100 * at1['_headway_wb4_seed0']:.0f}%",
         "decisions after 1 s, without and with the wait bias")
    card(s, x2, 3.85, w2, 2.72)
    text(s, x2 + 0.25, 4.0, w2 - 0.5, 0.3, [[("Honest limits", {"bold": True, "color": MAROON2, "size": 14})]])
    bullet_lines(s, x2 + 0.25, 4.45, w2 - 0.5, [
        "metres features: PPO still below the index",
        "UAH (28 trips): below the supervised agent",
        "labels are simulated types; real labelled dense traffic still missing"], step=0.62, size=11)
    footer(s, n)
    return s


def slide_trace(prs, crest, n):
    s = new_slide(prs)
    header(s, "3 · RL labeler", "Hand check: a normal driver the fixed index would flag", crest)
    md = open(os.path.join(ROOT, "docs", "hand_checks", "rl_decision_trace_headway_normal.md")).read()
    rows = [r.split("|")[1:-1] for r in md.splitlines() if re.match(r"^\| \d+ \|", r)]
    t = pd.DataFrame([[float(x) if re.match(r"^-?[\d.]+$", x.strip()) else x.strip() for x in r] for r in rows],
                     columns=["s", "speed", "accel", "gap", "wave", "ai", "ai10", "pw", "pc", "pn", "pa", "action"])
    cd = CategoryChartData()
    cd.categories = [str(int(v)) for v in t["s"]]
    cd.add_series("P(wait)", [round(v, 3) for v in t["pw"]])
    cd.add_series("P(normal)", [round(v, 3) for v in t["pn"]])
    cd.add_series("running AI / 100", [round(v / 100, 3) for v in t["ai"]])
    cd.add_series("cut off 42", [0.42] * len(t))
    chart_card(s, X0, 1.65, 7.0, 4.92, "Policy over time, vehicle normal_r0.35 (test seed)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = 1
    for ser, c in zip(ch.plots[0].series, [MAROON2, GOLD, "4A6FA5", GREY_BAR]):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(2.25)
        ser.smooth = False
    first = re.search(r"- AI = (.*)", md).group(1)
    dec = re.search(r"label (\w+) after (\d+) s", md)
    x2, w2 = X0 + 7.25, W - 7.25
    card(s, x2, 1.65, w2, 4.92)
    text(s, x2 + 0.25, 1.8, w2 - 0.5, 0.3, [[("First second by hand", {"bold": True, "color": MAROON2, "size": 14})]])
    text(s, x2 + 0.25, 2.2, w2 - 0.5, 1.4, [l[2:] for l in md.splitlines() if l.startswith("- ")][:4], size=10, color=INK)
    text(s, x2 + 0.25, 3.75, w2 - 0.5, 0.8, f"AI = {first.split('= ')[-1]}, code agrees", size=11, color=INK)
    text(s, x2 + 0.25, 4.45, w2 - 0.5, 1.9, f"AI stays above 42 (fixed index: aggressive). The agent waits, "
         f"P(normal) grows, and it labels {dec.group(1)} after {dec.group(2)} s: correct.", size=12, color=MAROON2, bold=True)
    footer(s, n)
    return s


def slide_us101(prs, crest, n):
    s = new_slide(prs)
    header(s, "4 · Simulation", "US-101 congestion needs a bottleneck", crest)
    sw = pd.read_csv(os.path.join(DATA, "us101_sweep.csv"))
    sw["main2_lanes"] = sw.get("main2_lanes", 5)
    best = lambda q: sw[q].sort_values("ks_speed").iloc[0]  # noqa: E731
    cases = [("IDM, no bottleneck", best((sw["model"] == "IDM") & (sw["main2_lanes"] == 5))),
             ("EIDM, no bottleneck", best((sw["model"] == "EIDM") & (sw["main2_lanes"] == 5) & (sw["main2_speed"] > 29))),
             ("Krauss, no bottleneck", best((sw["model"] == "Krauss") & (sw["main2_lanes"] == 5) & (sw["main2_speed"] > 29))),
             ("EIDM, lane drop", best((sw["model"] == "EIDM") & (sw["main2_lanes"] < 5)))]
    cd = CategoryChartData()
    cd.categories = [c[0] for c in cases]
    cd.add_series("speed KS", [round(float(c[1]["ks_speed"]), 3) for c in cases])
    cd.add_series("headway KS", [round(float(c[1]["ks_thw"]), 3) for c in cases])
    chart_card(s, X0, 1.65, 7.0, 4.92, "Distance to NGSIM US-101 (KS, lower is better)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [MAROON2, GOLD])
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = 1
    ch.plots[0].gap_width = 70
    b = cases[-1][1]
    x2, w2 = X0 + 7.25, W - 7.25
    stat(s, x2, 1.65, w2, f"{b['speed_med']:.0f}", " km/h", f"median speed, EIDM with {int(b['main2_lanes'])} lanes after the "
         f"weave at {int(b['main'])} veh/h (US-101: 48)", dark=True)
    card(s, x2, 2.75, w2, 3.82)
    bullet_lines(s, x2 + 0.25, 2.95, w2 - 0.5, [
        "IDM breaks into stop and go; EIDM and Krauss stay free flowing at any demand",
        "SUMO only inserts cars when there is room, so demand alone cannot jam the road",
        "a downstream lane drop makes the queue reach the measured section",
        "target speed KS < 0.2 not reached yet"], step=0.82, size=11)
    footer(s, n)
    return s


def slide_next(prs, crest, n):
    s = new_slide(prs)
    header(s, "Next", "Decisions for Dr. Daher and next steps", crest)
    card(s, X0, 1.65, 5.65, 4.92, fill=MAROON)
    text(s, X0 + 0.3, 1.85, 5.1, 0.3, [[("Decisions", {"bold": True, "color": "FFFFFF", "size": 16})]])
    bullet_lines(s, X0 + 0.3, 2.35, 5.0, [
        "Adopt the fitted proximity mix as the reference index (refit cut offs, rerun every result)?",
        "RL direction: continue the sequential labeler (option a) or move to inverse RL on highD for the paper?",
        "Paper target: IEEE ITSC, outline ready (docs/paper_outline.md)"], color=PALE, marker=PINK, step=1.05)
    x2 = X0 + 5.9
    card(s, x2, 1.65, W - 5.9, 4.92)
    text(s, x2 + 0.3, 1.85, W - 6.5, 0.3, [[("Next steps", {"bold": True, "color": MAROON2, "size": 16})]])
    bullet_lines(s, x2 + 0.3, 2.35, W - 6.6, [
        "highD and exiD (downloading): score with the mix, density effect on German motorways",
        "Full DriveDNA (downloading): radar car following, 465 drivers",
        "RL labeler on real data once labelled dense traffic exists",
        "US-101 calibration: finer demand around the lane drop"], step=0.7)
    footer(s, n)
    return s


def main():
    prs = Presentation(B.SRC)
    with zipfile.ZipFile(B.SRC) as z:
        tmp = tempfile.mkdtemp()
        crest, logo = os.path.join(tmp, "crest.png"), os.path.join(tmp, "logo.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
        open(logo, "wb").write(z.read("ppt/media/image-1-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):                      # start from the September deck's master, without its slides
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    title_slide(prs, logo)
    for i, b in enumerate([slide_correction, slide_mix, slide_mdp, slide_rl_results, slide_trace, slide_us101, slide_next]):
        b(prs, crest, 2 + i)
    assert len(prs.slides) == B.TOTAL, len(prs.slides)
    prs.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    main()
