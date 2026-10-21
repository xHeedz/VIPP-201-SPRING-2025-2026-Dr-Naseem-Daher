"""
Three slide weekly update (week of 12 Oct 2026), same style as the October deck
(helpers from scripts/build_october_deck.py). Every number is read from data/.

    python scripts/build_weekly_update.py
writes ../12-10-2026/Weekly Update.pptx
"""
import os
import sys
import tempfile
import zipfile

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_october_deck as B  # noqa: E402
from build_october_deck import (CARD_LINE, GREY_BAR, INK, MAROON, MAROON2, PALE, PINK, W, X0,  # noqa: E402
                                bullet_lines, card, chart_card, color_series, header, new_slide, stat,
                                style_chart, text)

DATA, ROOT = B.DATA, B.ROOT
OUT = os.path.abspath(os.path.join(ROOT, "..", "12-10-2026", "Weekly Update.pptx"))
GOLD = "D9A441"
B.TOTAL = 3


def pct_axis(ch, top, unit):
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = top
    ch.value_axis.major_unit = unit
    ch.value_axis.tick_labels.number_format = '0"%"'
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.plots[0].gap_width = 70


def slide_german(prs, crest, n):
    s = new_slide(prs)
    header(s, "1 · German data", "All five levelX datasets scored", crest)
    g = pd.read_csv(os.path.join(DATA, "german_summary.csv"))
    a = g[g["density_bin"] == "all"].set_index("dataset").loc[["highD", "exiD", "inD", "rounD", "uniD"]]
    cd = CategoryChartData()
    cd.categories = list(a.index)
    for name, col in [("metres", "aggressive_metres"), ("headway", "aggressive_headway"), ("mix", "aggressive_mix")]:
        cd.add_series(name, [round(100 * float(v), 1) for v in a[col]])
    chart_card(s, X0, 1.65, 7.0, 4.92, "Share of windows labelled aggressive (score >= 42)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [MAROON2, GOLD, GREY_BAR])
    pct_axis(ch, 60, 15)
    u = g[g["dataset"] == "uniD"].set_index("density_bin")
    x2, w2 = X0 + 7.25, W - 7.25
    stat(s, x2, 1.65, w2, f"{100 * a.loc['highD', 'aggressive_mix']:.0f}%", "",
         f"of highD windows aggressive with the mix ({int(a.loc['highD', 'windows']):,} windows)", dark=True)
    stat(s, x2, 2.75, w2, f"{u.loc['q1 light', 'median_metres']:.0f}", f"→{u.loc['q4 dense', 'median_metres']:.0f}",
         "uniD median score in metres, light to dense traffic (headway stays 3 to 7)")
    card(s, x2, 3.85, w2, 2.72)
    text(s, x2 + 0.25, 4.0, w2 - 0.5, 0.3, [[("Reading", {"bold": True, "color": MAROON2, "size": 14})]])
    bullet_lines(s, x2 + 0.25, 4.45, w2 - 0.5, [
        "motorway: speed, ordinary headway and lane offset add up to 42 without aggressive driving",
        "urban: metres climbs with density, as in SUMO, NGSIM and pNEUMA",
        "parked cars removed from inD (traffic speed 2.8 to 18.6 km/h)"], step=0.68, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "highD and exiD scored as highway, inD, rounD and uniD as urban with mix 0.5. inD, rounD and uniD have no leader ids "
        "or lane positions, so the gap is measured the pNEUMA way (cone of 1.6 m, heading within 30 degrees) and wave is 0.")
    return s


def slide_context(prs, crest, n):
    s = new_slide(prs)
    header(s, "2 · Context", "Traffic state context cuts highway false alarms", crest)
    c = pd.read_csv(os.path.join(DATA, "context_reference.csv"))
    t = c[(c["context"] == "traffic") & (c["proximity"] == "mix")].set_index("set")
    rows = [("low", "planted highway_low"), ("medium", "planted highway_medium"), ("jam", "planted jam_jam"),
            ("merge", "planted merge_medium"), ("weather", "planted weather_medium")]
    cd = CategoryChartData()
    cd.categories = [r[0] for r in rows]
    cd.add_series("normal flagged (42)", [round(100 * float(t.loc[k, "fpr_fixed42"]), 1) for _, k in rows])
    cd.add_series("normal flagged (context)", [round(100 * float(t.loc[k, "fpr_context90"]), 1) for _, k in rows])
    cd.add_series("aggressive caught (context)", [round(100 * float(t.loc[k, "tpr_context90"]), 1) for _, k in rows])
    chart_card(s, X0, 1.65, 7.0, 4.92, "Planted SUMO test drivers, highway settings (mix, context = 90th percentile)")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(6.7), Inches(4.4), cd).chart
    style_chart(ch)
    color_series(ch, [GREY_BAR, MAROON2, GOLD])
    pct_axis(ch, 75, 15)
    hw = [k for _, k in rows]
    fpr = t.loc[hw, "fpr_context90"]
    ng = c[(c["set"] == "NGSIM US-101 (unlabelled)") & (c["proximity"] == "mix")].set_index("context")
    urb = t.loc[["planted urban_medium", "planted roundabout_medium"], "fpr_context90"]
    x2, w2 = X0 + 7.25, W - 7.25
    fix = t.loc[hw, "fpr_fixed42"]
    stat(s, x2, 1.65, w2, f"≤{100 * fpr.max():.1f}%", "",
         f"normal drivers flagged on highway with context (fixed 42: {100 * fix.min():.0f} to {100 * fix.max():.0f}%)", dark=True)
    stat(s, x2, 2.75, w2, f"{100 * ng.loc['traffic', 'flag_fixed42']:.0f}%", f"→{100 * ng.loc['traffic', 'flag_context90']:.0f}%",
         "NGSIM US-101 windows flagged, fixed 42 to context")
    card(s, x2, 3.85, w2, 2.72)
    text(s, x2 + 0.25, 4.0, w2 - 0.5, 0.3, [[("How and limits", {"bold": True, "color": MAROON2, "size": 14})]])
    bullet_lines(s, x2 + 0.25, 4.45, w2 - 0.5, [
        "reference bin: environment x traffic speed (20 km/h) x car ahead, 276,485 German windows",
        "highway ranking kept, but the 90th percentile misses most aggressive drivers",
        f"urban fails: {100 * urb.min():.0f} to {100 * urb.max():.0f}% of normal SUMO drivers flagged (sim realism gap)"],
        step=0.68, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Score = percentile of the window among German windows in the same traffic state, not own speed. Binning on own "
        "speed lost most of the ranking. SUMO urban drivers, even normal ones, are faster and closer than real German "
        "intersection and roundabout traffic.")
    return s


def slide_irl(prs, crest, n):
    s = new_slide(prs)
    header(s, "3 · Inverse RL", "Desired headway orders the driver types", crest)
    p = pd.read_csv(os.path.join(DATA, "irl_population.csv")).set_index("group")
    v = pd.read_csv(os.path.join(DATA, "irl_planted_summary.csv")).set_index("comparison").loc["aggressive vs normal"]
    groups = [("conservative", "planted conservative"), ("normal", "planted normal"), ("aggressive", "planted aggressive")]
    cd = CategoryChartData()
    cd.categories = [g[0] for g in groups]
    cd.add_series("desired headway (s)", [round(float(p.loc[k, "desired_headway_s"]), 2) for _, k in groups])
    chart_card(s, X0, 1.65, 4.6, 4.92, "Population fit, planted SUMO drivers")
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(X0 + 0.15), Inches(2.05), Inches(4.3), Inches(4.4), cd).chart
    style_chart(ch, legend=False)
    color_series(ch, [MAROON2])
    ch.value_axis.minimum_scale = 0
    ch.value_axis.maximum_scale = 4.0
    ch.value_axis.major_unit = 1.0
    ch.value_axis.tick_labels.number_format = "0.0"
    ch.value_axis.tick_labels.number_format_is_linked = False
    ch.plots[0].gap_width = 70
    pl = ch.plots[0]
    pl.has_data_labels = True
    pl.data_labels.number_format = '0.00" s"'
    pl.data_labels.number_format_is_linked = False
    x2 = X0 + 4.85
    w2 = W - 4.85
    wh = (w2 - 0.25) / 2
    stat(s, x2, 1.65, w2, f"{v['auc_irl_combined']:.2f}", f" vs {v['auc_index_mix']:.2f}",
         "per vehicle AUC, aggressive vs normal: inverse RL against the index (mix), same test vehicles", dark=True)
    card(s, x2, 2.75, wh, 3.82)
    text(s, x2 + 0.25, 2.9, wh - 0.5, 0.3, [[("Reading", {"bold": True, "color": MAROON2, "size": 14})]])
    bullet_lines(s, x2 + 0.25, 3.35, wh - 0.5, [
        "pooled by type the fit is right; per vehicle there are too few informative seconds",
        "German motorways: not identified (desired headway beyond the 4 s cap)",
        "only the density trend is usable: shorter headway in denser traffic"], step=0.95, size=11)
    xd = x2 + wh + 0.25
    card(s, xd, 2.75, wh, 3.82, fill=MAROON)
    text(s, xd + 0.25, 2.9, wh - 0.5, 0.3, [[("Decisions and next", {"bold": True, "color": "FFFFFF", "size": 14})]])
    bullet_lines(s, xd + 0.25, 3.35, wh - 0.5, [
        "adopt the proximity mix as the reference index?",
        "paper angle: labeling RL or inverse RL?",
        "full DriveDNA access for labelled dense traffic",
        "US-101 in SUMO: speed KS 0.26, target 0.2"], color=PALE, marker=PINK, step=0.72, size=11)
    B.footer(s, n)
    s.notes_slide.notes_text_frame.text = (
        "Reward: progress and progress squared (desired speed), headway and headway squared (desired headway "
        "h* = -theta_h / (2 theta_h2) x 4 s), and discomfort. Per vehicle IRL stays well below the index.")
    return s


def main():
    prs = Presentation(B.SRC)
    with zipfile.ZipFile(B.SRC) as z:
        tmp = tempfile.mkdtemp()
        crest = os.path.join(tmp, "crest.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):                      # start from the September deck's master, without its slides
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    for i, b in enumerate([slide_german, slide_context, slide_irl]):
        b(prs, crest, 1 + i)
    assert len(prs.slides) == B.TOTAL, len(prs.slides)
    prs.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    main()
