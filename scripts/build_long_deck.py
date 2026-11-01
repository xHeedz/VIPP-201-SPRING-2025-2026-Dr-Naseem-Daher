"""
Long progress presentation (week of 26 Oct 2026): every slide is a visual (native chart, table, diagram, video or
animation), the explanation is in each slide's notes. Fade transitions on every slide; diagrams build on click.
Every number is read from data/.

    python scripts/build_long_deck.py
writes ../26-10-2026/Progress Presentation.pptx
"""
import json
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import DATA_DIR, FIG_DIR  # noqa: E402

ROOT = os.path.abspath(os.path.join(DATA_DIR, ".."))
OUT_DIR = os.path.abspath(os.path.join(ROOT, "..", "26-10-2026"))
OUT = os.path.join(OUT_DIR, "Progress Presentation.pptx")
AUDIT = os.path.join(OUT_DIR, "label_audit")
GIFS = os.path.abspath(os.path.join(ROOT, "..", "20-10-2026"))
SRC = os.path.abspath(os.path.join(ROOT, "..", "28:9:2026", "Presentation", "VIPP 201A Session Results.pptx"))

# ── design system ───────────────────────────────────────────────────────────
MAROON, DEEP, GOLD, INK, MUTED, TINT, LINE, WHITE = "7A1F2B", "4E0F1A", "C99A2E", "222222", "6B6B6B", "F6F0F1", "E3DADC", "FFFFFF"
BLUE, GREY = "4A6FA5", "A6A6A6"
HEAD, BODY = "Georgia", "Arial"
NUM = {"font": "Arial", "bold": True}
SW, SH = 13.333, 7.5
MX = 0.6                       # side margin
TOTAL = [0]


def rgb(h):
    return RGBColor.from_string(h)


def tb(s, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
       name=None, italic=False):
    """runs: str or list of paragraphs (str or list of (text, opts))"""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(6)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name, f.size = o.get("font", font), Pt(o.get("size", size))
            f.bold, f.italic = o.get("bold", bold), o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return box


def rect(s, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None, line=None):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        r.adjustments[0] = radius
    r.fill.solid()
    r.fill.fore_color.rgb = rgb(fill)
    if line:
        r.line.color.rgb = rgb(line)
        r.line.width = Pt(1)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    if name:
        r.name = name
    return r


def labelled(s, x, y, w, h, t, fill=MAROON, color=WHITE, size=15, bold=True, name=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    r = rect(s, x, y, w, h, fill, shape=shape, name=name)
    tf = r.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(t.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        rr = p.add_run()
        rr.text = line
        rr.font.name, rr.font.size, rr.font.bold = BODY, Pt(size if i == 0 else size - 3), bold if i == 0 else False
        rr.font.color.rgb = rgb(color)
    return r


def circle_num(s, x, y, n, d=0.5, fill=MAROON):
    return labelled(s, x, y, d, d, str(n), fill=fill, size=16, shape=MSO_SHAPE.OVAL)


def arrow(s, x1, y1, x2, y2, color=MAROON, name=None):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(2.5)
    c.line._get_or_add_ln().append(etree.fromstring(
        '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'))
    if name:
        c.name = name
    return c


def new(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[0])
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DEEP if dark else WHITE)
    return s


def title(s, kicker, text, n, crest):
    tb(s, MX, 0.42, 10, 0.3, kicker.upper(), size=11, color=MAROON, bold=True)
    tb(s, MX, 0.72, 11.2, 0.75, [[(text, {"font": HEAD, "size": 30, "color": INK})]])
    s.shapes.add_picture(crest, Inches(SW - MX - 0.42), Inches(0.42), Inches(0.42))
    tb(s, SW - MX - 1.0, SH - 0.45, 1.0, 0.3, f"{n} / {TOTAL[0]}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def bullets(s, x, y, w, items, size=15, color=INK, dot=MAROON, step=0.55):
    """short takeaway lines: a dot and one line each, no paragraphs"""
    for i, t in enumerate(items):
        yy = y + i * step
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(yy + 0.09), Inches(0.13), Inches(0.13))
        d.fill.solid()
        d.fill.fore_color.rgb = rgb(dot)
        d.line.fill.background()
        tb(s, x + 0.28, yy, w - 0.28, step, t, size=size, color=color)


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def style_chart(ch, size=12, legend=True):
    ch.font.name, ch.font.size = BODY, Pt(size)
    ch.font.color.rgb = rgb(INK)
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)


def table(s, x, y, w, h, rows, col_w=None, header_fill=MAROON, size=14, cell_fill=None, bold_first_col=True, left_cols=None):
    """rows: list of lists of str; cell_fill: function (r, c) -> hex or None"""
    nr, nc = len(rows), len(rows[0])
    t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h)).table
    tbl = t._tbl
    tbl.tblPr.set("bandRow", "0")
    style = tbl.tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"           # no style, no grid
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(h / nr)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.12)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = header_fill if i == 0 else (cell_fill(i, j) if cell_fill else None) or (TINT if i % 2 == 0 else WHITE)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(fill)
            tf = c.text_frame
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if (j == 0 or left_cols and j in left_cols) else PP_ALIGN.CENTER
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            dark = i == 0 or fill in (MAROON, DEEP, BLUE)
            r.font.name, r.font.size = BODY, Pt(size)
            r.font.bold = i == 0 or (j == 0 and bold_first_col)
            r.font.color.rgb = rgb(WHITE if dark else INK)
    return t


def heat(v, lo, hi):
    """white -> maroon tint scale for a value between lo and hi"""
    a = float(np.clip((v - lo) / (hi - lo), 0, 1))
    base, top = np.array([255, 255, 255]), np.array([0xC9, 0x7A, 0x84])
    c = (base + a * (top - base)).astype(int)
    return "%02X%02X%02X" % tuple(c)


# ── animations (slide XML) ──────────────────────────────────────────────────
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def add_transition(s):
    sld = s._element
    tr = etree.SubElement(sld, qn("p:transition"), spd="med")
    etree.SubElement(tr, qn("p:fade"))
    timing = sld.find(qn("p:timing"))
    if timing is not None:
        sld.remove(timing)
        sld.append(timing)


def add_builds(s, groups):
    """groups: list of lists of shape ids; each group fades in on one click"""
    ids = iter(range(3, 10000))
    pars = []
    for grp in groups:
        inner = []
        for k, sid in enumerate(grp):
            a, b, c2 = next(ids), next(ids), next(ids)
            inner.append(
                f'<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" '
                f'nodeType="{"clickEffect" if k == 0 else "withEffect"}"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
                f'<p:childTnLst><p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/>'
                f'</p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{sid}"/></p:tgtEl><p:attrNameLst><p:attrName>'
                f'style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
                f'<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{c2}" dur="500"/><p:tgtEl>'
                f'<p:spTgt spid="{sid}"/></p:tgtEl></p:cBhvr></p:animEffect></p:childTnLst></p:cTn></p:par>')
        o1, o2 = next(ids), next(ids)
        pars.append(f'<p:par><p:cTn id="{o1}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>'
                    f'<p:childTnLst><p:par><p:cTn id="{o2}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
                    f'<p:childTnLst>{"".join(inner)}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>')
    bld = "".join(f'<p:bldP spid="{sid}" grpId="0"/>' for grp in groups for sid in grp)
    xml = (f'<p:timing xmlns:p="{P}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
           f'<p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">'
           f'<p:childTnLst>{"".join(pars)}</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl>'
           f'<p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl>'
           f'<p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>'
           f'<p:bldLst>{bld}</p:bldLst></p:timing>')
    s._element.append(etree.fromstring(xml))


# ── data ────────────────────────────────────────────────────────────────────
def j(name):
    return json.load(open(os.path.join(DATA_DIR, name)))


def csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


# ── slides ──────────────────────────────────────────────────────────────────
SLIDES = []


def slide(fn):
    SLIDES.append(fn)
    return fn


@slide
def s_title(prs, crest, logo, n):
    s = new(prs, dark=True)
    s.shapes.add_picture(logo, Inches(MX), Inches(0.55), Inches(3.0))
    tb(s, MX, 2.35, 11, 0.4, "VIPP 301A  ·  PROGRESS PRESENTATION", size=13, color="E8C9CE", bold=True)
    tb(s, MX, 2.85, 11.5, 1.1, [[("Driver Aggressiveness Index", {"font": HEAD, "size": 48, "color": WHITE})]])
    tb(s, MX, 4.05, 11.5, 0.6, [[("label audit  ·  new signals  ·  KAN  ·  German roads",
                                  {"size": 20, "color": "E8C9CE"})]])
    tb(s, MX, 6.3, 8, 0.6, [[("Hadi Al Shmaissani", {"bold": True, "color": WHITE, "size": 16})],
                            [("Supervisor: Dr. Naseem Daher  ·  American University of Beirut", {"color": "E8C9CE", "size": 13})]])
    notes(s, "Since the last meeting: a manual check of the labels, new signals (time to collision, speed relative to the "
             "limit and to the traffic, a physics based residual), a network that learns the formula, how long the agents need "
             "to train, scenario experts switched by a state machine, and moving the model to German roads.")
    return s


@slide
def s_pipeline(prs, crest, logo, n):
    s = new(prs)
    title(s, "Overview", "Pipeline", n, crest)
    steps = [("Data", "UAH · SUMO\nNGSIM · highD"), ("Cleaning", "unknown stays\nunknown"),
             ("Signals", "TTC · relative speed\nIDM residual"), ("Score", "formula · KAN"), ("Agent", "PPO · DQN · FSM")]
    w, gap, y = 2.05, 0.42, 2.6
    groups = []
    for i, (t, sub) in enumerate(steps):
        x = MX + i * (w + gap)
        c = circle_num(s, x + w / 2 - 0.25, y - 0.75, i + 1)
        b = labelled(s, x, y, w, 1.55, f"{t}\n{sub}", fill=MAROON if i >= 2 else TINT, color=WHITE if i >= 2 else MAROON, size=18)
        grp = [c.shape_id, b.shape_id]
        if i:
            a = arrow(s, x - gap + 0.04, y + 0.78, x - 0.04, y + 0.78)
            grp.append(a.shape_id)
        groups.append(grp)
    out = labelled(s, MX + 4 * (w + gap), 5.0, w, 1.0, "label\nconservative · normal · aggressive", fill=WHITE, color=MAROON,
                   size=15)
    out.line.color.rgb = rgb(MAROON)
    a = arrow(s, MX + 4 * (w + gap) + w / 2, y + 1.57, MX + 4 * (w + gap) + w / 2, 4.98)
    groups.append([out.shape_id, a.shape_id])
    bullets(s, MX, 4.9, 8.8, ["same cleaning for every dataset", "unknown values are never treated as 0",
                              "agents decide when they have seen enough of a driver"])
    notes(s, "The whole pipeline. Every dataset goes through the same cleaning, where an unknown value stays unknown instead of "
             "becoming zero. New signals are computed from the cleaned data. A score comes from the fixed formula or from the "
             "KAN, which learns the formula. The agents watch a driver second by second and decide when to give the label. "
             "Each step appears on click.")
    return s, groups


@slide
def s_equations(prs, crest, logo, n):
    s = new(prs)
    title(s, "Model", "Equations", n, crest)
    w = j("extended_weights.json")["weights"]
    rows = [["model", "equation"],
            ["Reference", "AI = 100 · Σ wₖ xₖ / Σ wₖ   (known terms only)"],
            ["Extended", f"AI = 100 · ({w[0]:.1f}v + {w[1]:.1f}a² + {w[2]:.1f}g² + {w[3]:.1f}ℓ + {w[4]:.1f}TTC + {w[5]:.1f}v_lim + {w[7]:.1f}IDM)"],
            ["KAN", "AI = σ( b + Σᵢ φᵢ(xᵢ) ),   φᵢ = wᵢ silu(x) + Σⱼ cᵢⱼ Bⱼ(x)"],
            ["TTC", "TTC = gap / closing speed,   term (1 − TTC/4)²"],
            ["IDM", "a* = a_max[1 − (v/v₀)⁴ − (s*/s)²],   residual = a − a*"],
            ["Risk reward", "−0.002(1 + 5dₜ) per wait,   −(1 + 2 max dₜ) per missed aggressive"]]
    table(s, MX, 1.7, SW - 2 * MX, 4.25, rows, col_w=[2.3, SW - 2 * MX - 2.3], size=16, left_cols={1})
    bullets(s, MX, 6.15, SW - 2 * MX, ["unknown terms are left out and the other weights rescaled",
                                         "danger d = TTC under 4 s or headway under 1 s"], size=14, step=0.45)
    notes(s, "Reference: weighted mean of the known terms (v speed / 150, a acceleration / 5, g gap closeness, l lane offset / "
             f"1.5 m). Extended: four more terms, fitted weights {', '.join(f'{x:.1f}' for x in w)} (speed, acceleration², gap, "
             "lane, TTC, speed vs limit, speed vs traffic, IDM). KAN: one learned curve per input (cubic B-splines, grid 5), "
             "summed, then a sigmoid. TTC: seconds to a collision at the current speeds. IDM: the acceleration a calm driver "
             "would choose (a_max 1.5, b 2.0, T 1.5 s, s0 2 m, v0 the limit); the residual is the gap between the real driver and "
             "that. Risk reward: d is the danger of a second (TTC under 4 s or headway under 1 s).")
    return s


@slide
def s_audit_matrix(prs, crest, logo, n):
    s = new(prs)
    title(s, "Label audit", "Formula against label on UAH", n, crest)
    a = csv("label_audit_summary.csv").set_index("group")
    nn, na = a.loc["agree: normal", "windows"], a.loc["label normal, formula aggressive", "windows"]
    ac, aa = a.loc["label aggressive, formula calm", "windows"], a.loc["agree: aggressive", "windows"]
    rows = [["", "formula: calm", "formula: aggressive"],
            ["label: normal", f"{nn:,}", f"{na:,}"],
            ["label: aggressive", f"{ac:,}", f"{aa:,}"]]
    fill = {(1, 1): "EADFE1", (2, 2): "EADFE1", (1, 2): heat(0.9, 0, 1), (2, 1): heat(0.9, 0, 1)}
    table(s, MX, 2.0, 6.4, 3.0, rows, col_w=[2.2, 2.1, 2.1], size=20, cell_fill=lambda i, k: fill.get((i, k)))
    bullets(s, MX, 5.3, 6.6, ["label aggressive, formula calm: the label is wrong",
                              "label normal, formula aggressive: just over the cut-off",
                              "manual check: those drivers are not aggressive"], size=14, step=0.48)
    side = a.loc["label aggressive, formula calm", "drivesafe_sides_with_formula"]
    cd = CategoryChartData()
    cd.categories = ["agrees with the formula", "agrees with the label"]
    cd.add_series("UAH software", [round(100 * side, 1), round(100 * (1 - side), 1)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.DOUGHNUT, Inches(7.4), Inches(1.55), Inches(5.3), Inches(4.5), cd)
    ch = gf.chart
    style_chart(ch)
    plot = ch.plots[0]
    for i, c in enumerate((MAROON, GREY)):
        pt = plot.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    plot.has_data_labels = True
    plot.data_labels.number_format = '0"%"'
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.font.size, plot.data_labels.font.color.rgb = Pt(14), rgb(WHITE)
    ch.has_title = True
    ch.chart_title.text_frame.text = "label aggressive, formula calm: who does UAH's software side with?"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    tb(s, 7.4 + 5.3 / 2 - 1.0, 3.45, 2.0, 0.9, [[(f"{side:.0%}", {**NUM, "size": 34, "color": MAROON})],
                                               [("with the formula", {"size": 11, "color": MUTED})]], align=PP_ALIGN.CENTER)
    notes(s, f"Every UAH 10 second window: the trip label against the formula. Shaded cells are disagreements: {ac} windows "
             f"labelled aggressive that the formula calls calm, and {na} labelled normal that it calls aggressive. For the first "
             f"group UAH's own software (DriveSafe) is a third opinion: it sides with the formula in {side:.0%} of them, so most "
             "are calm stretches inside an aggressive trip and the label is wrong. DriveSafe never flags a second of a normal "
             "trip, so it cannot judge the second group. Manual check (videos): those drivers are not aggressive; the 8 sampled "
             "windows scored only 27.7 to 31.7, just over the cut-off of 27.5, and 79% of the 697 sit within 5 points of it. "
             "So the first group is mostly wrong labels, the second mostly formula false alarms at the edge of the cut-off.")
    return s


@slide
def s_audit_cases(prs, crest, logo, n):
    s = new(prs)
    title(s, "Label audit", "Two sampled windows", n, crest)
    w = (SW - 2 * MX - 0.4) / 2
    for i, (f, cap) in enumerate((("02.png", "gap closes to 10 m: label right, formula late → TTC"),
                                  ("10.png", "100 km/h in a 120 zone: formula wrong → speed vs limit"))):
        x = MX + i * (w + 0.4)
        p = s.shapes.add_picture(os.path.join(AUDIT, "figures", f), Inches(x), Inches(1.6), Inches(w))
        circle_num(s, x - 0.15, 1.45, i + 1, d=0.45)
        bullets(s, x, 1.6 + p.height / 914400 + 0.15, w, [cap], size=14)
    notes(s, "Left: labelled aggressive, the formula says calm. Speed at the limit, gentle acceleration, but the gap closes from "
             "50 m to 10 m at the end of the window and the time to collision drops to about one second: the label is probably "
             "right and the formula late (the closing in only shows partly in a 10 s average). Right: labelled normal, the "
             "formula says aggressive. 100 km/h in a 120 km/h zone behind trucks: the label is right, raw speed misled the "
             "formula. So the audit finds both directions.")
    return s


@slide
def s_audit_video(prs, crest, logo, n):
    s = new(prs)
    title(s, "Label audit", "The same two windows on video", n, crest)
    w = (SW - 2 * MX - 0.4) / 2
    h = w * 9 / 16
    for i, c in enumerate(("02", "10")):
        x = MX + i * (w + 0.4)
        s.shapes.add_movie(os.path.join(AUDIT, "clips", f"{c}.mp4"), Inches(x), Inches(1.75), Inches(w), Inches(h),
                           poster_frame_image=os.path.join(AUDIT, "clips", f"{c}.mp4.png"), mime_type="video/mp4")
        circle_num(s, x - 0.15, 1.6, i + 1, d=0.45)
        bullets(s, x, 1.9 + h, w, [["labelled aggressive · closing in at the end", "labelled normal · ordinary motorway driving"][i]],
                size=14)
    notes(s, "The road video of the two windows, 10 s each, synchronised the way UAH's reader tool does it. All 22 sampled "
             "windows are in the label_audit folder (index.html), with review.csv for the manual verdict.")
    return s


@slide
def s_smoothing(prs, crest, logo, n):
    s = new(prs)
    title(s, "Label audit", "From a hard cut-off to a gradual probability", n, crest)
    sys.path.insert(0, ROOT)
    from datasets.windows import TERM_COLS
    from model.aggressiveness_model import REF_CUTOFFS, reference_score
    w = pd.read_csv(os.path.join(DATA_DIR, "clean", "windows_uah.csv.gz"))
    w = w[w["label"].isin(["normal", "aggressive"])]
    sc = reference_score(w[TERM_COLS].to_numpy())
    edges = [0, 15, 20, 25, 27.5, 30, 35, 40, 100]
    names = ["<15", "15-20", "20-25", "25-27.5", "27.5-30", "30-35", "35-40", ">40"]
    share = (w["label"] == "aggressive").groupby(pd.cut(sc, edges)).mean().to_numpy()
    near = float(((np.abs(sc - REF_CUTOFFS[1]) <= 2) | (np.abs(sc - REF_CUTOFFS[0]) <= 2)).mean())
    cd = CategoryChartData()
    cd.categories = names
    cd.add_series("share from aggressive trips", [round(100 * v, 1) for v in share])
    cd.add_series("hard cut-off label", [0, 0, 0, 0, 100, 100, 100, 100])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(MX), Inches(1.55), Inches(8.0), Inches(5.0), cd)
    ch = gf.chart
    style_chart(ch, size=12)
    for ser, c, wdt in zip(ch.plots[0].series, (MAROON, GREY), (3.25, 2.0)):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(wdt)
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(c)
        ser.marker.format.line.color.rgb = rgb(c)
        ser.smooth = False
    ch.plots[0].series[1].format.line.dash_style = 4
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, 100, 25
    va.major_gridlines.format.line.color.rgb = rgb(LINE)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = '0"%"', False
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "formula score (UAH 10 s windows)"
    ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    tb(s, 9.0, 1.7, 3.8, 1.3, [[(f"{near:.0%}", {**NUM, "size": 40, "color": MAROON})],
                               [("of windows within 2 points of a cut-off", {"size": 14})]])
    bullets(s, 9.0, 3.35, 3.8, ["probability instead of a label", "borderline band: keep watching",
                                "hysteresis: switch only after N seconds", "small FSM, like a thermostat"], size=14, step=0.62)
    notes(s, "Why the hard cut-off is a problem: two drivers who behave almost the same can land on opposite sides of 27.5 "
             f"and get opposite labels; {near:.0%} of UAH windows are within 2 points of a cut-off. The maroon line is the real "
             "share of windows from aggressive trips at each score: it rises gradually (about a quarter just under 27.5, a third "
             "just over it, 93% from 35 to 40), not in one jump like the grey cut-off line. Three ways to smooth it: report a "
             "probability (a sigmoid fitted to this curve), keep a borderline band where the answer is 'not sure yet, keep "
             "watching' (the agent's wait button already does this), and hysteresis over time: only switch to aggressive after "
             "staying above an upper threshold for several seconds and back to normal only below a lower one, a small state "
             "machine like a thermostat.")
    return s


@slide
def s_smoothing_result(prs, crest, logo, n):
    s = new(prs)
    title(s, "Label audit", "What smoothing does to the disagreements", n, crest)
    w = pd.read_csv(os.path.join(DATA_DIR, "smoothing_windows.csv"))
    fa = w[w["audit_group"] == "label normal, formula aggressive"]
    fc = w[w["audit_group"] == "label aggressive, formula calm"]
    y = w["label"] == "aggressive"
    sure = w["band"] != "borderline"
    rows = [["method", f"false alarms ({len(fa)})\nstill aggressive", f"missed ({len(fc)})\nstill calm", "accuracy, all UAH"]]
    rows.append(["hard cut-off 27.5", f"{len(fa)}", f"{len(fc)}", f"{((w['hard'] == 'aggressive') == y).mean():.1%}"])
    rows.append(["probability over 50%", f"{(fa['prob'] >= 0.5).sum()}", f"{(fc['prob'] < 0.5).sum()}",
                 f"{((w['prob'] >= 0.5) == y).mean():.1%}"])
    rows.append(["borderline band 25 to 30", f"{(fa['band'] == 'aggressive').sum()}", f"{(fc['band'] == 'normal').sum()}",
                 f"{((w.loc[sure, 'band'] == 'aggressive') == y[sure]).mean():.1%}*"])
    rows.append(["hysteresis (2 windows)", f"{(fa['hyst'] == 'aggressive').sum()}", f"{(fc['hyst'] == 'normal').sum()}",
                 f"{((w['hyst'] == 'aggressive') == y).mean():.1%}"])
    table(s, MX, 1.6, 8.3, 3.6, rows, col_w=[2.6, 2.0, 1.8, 1.9], size=14, left_cols=[0],
          cell_fill=lambda r, c: "EBD3A8" if (r, c) in ((2, 1), (3, 2), (3, 3)) else None)
    tb(s, MX, 5.3, 8.3, 0.4, f"* on the {sure.mean():.0%} of windows outside the band; the rest wait for more data",
       size=11, color=MUTED)
    moved = int((fa["band"] == "borderline").sum())
    tb(s, 9.3, 1.6, 3.5, 1.3, [[(f"{moved}", {**NUM, "size": 44, "color": MAROON})],
                               [(f"of {len(fa)} false alarms move to borderline", {"size": 14})]])
    bullets(s, 9.3, 3.3, 3.5, ["probability: fewest false alarms", "band: best accuracy, waits on 24%", "50/50 point on UAH: 30.3, not 27.5",
                               f"hysteresis also catches {int((fc['hyst'] == 'aggressive').sum())} missed"], size=14, step=0.55)
    return s


@slide
def s_signals(prs, crest, logo, n):
    s = new(prs)
    title(s, "New signals", "What each signal is worth on its own", n, crest)
    t = csv("extended_single_terms.csv").set_index("test")
    cols = [("idm", "IDM residual (physics)"), ("gap2", "gap"), ("accel2", "acceleration²"), ("speed", "speed"),
            ("rel_limit", "speed vs limit"), ("rel_traffic", "speed vs traffic"), ("lane", "lane offset"), ("ttc", "time to collision")]
    rows = [["signal", "UAH (real)", "SUMO (simulated)"]]
    vals = {}
    for i, (c, nm) in enumerate(cols, 1):
        u, p = t.loc["UAH A vs N", c], t.loc["planted test A vs N", c]
        rows.append([nm, "n/a" if pd.isna(u) else f"{u:.2f}", f"{p:.2f}"])
        vals[(i, 1)], vals[(i, 2)] = u, p
    table(s, MX, 1.7, 8.0, 4.9, rows, col_w=[3.6, 2.2, 2.2], size=16,
          cell_fill=lambda i, k: heat(vals[(i, k)], 0.5, 0.9) if (i, k) in vals and pd.notna(vals[(i, k)]) else None)
    tb(s, 9.1, 1.8, 3.7, 1.2, [[("0.78", {**NUM, "size": 44, "color": MAROON})],
                               [("physics residual on real drivers", {"size": 14, "color": INK})]])
    bullets(s, 9.1, 3.5, 3.7, ["AUC of one signal alone", "0.5 = coin flip, darker = better",
                               "TTC alone: coin flip over 10 s", "so TTC moved into the reward"], size=14, step=0.6)
    notes(s, "Each row is one signal used alone to separate aggressive from normal (0.5 is a coin flip, darker is better). The "
             "physics residual (IDM) is among the two best on real UAH drivers and the best on the simulated ones; on the "
             "simulation this is partly by construction (the simulated drivers are IDM drivers), so the real data result is the "
             "convincing one. Time to collision alone is a coin flip at the 10 second level: closing in moments are short and "
             "rare and vanish in the average, so TTC is used as a danger signal in the reward instead.")
    return s


@slide
def s_radar(prs, crest, logo, n):
    s = new(prs)
    title(s, "Learned formula", "Reference, extended formula and KAN on each UAH driver", n, crest)
    e = csv("extended_fit.csv")
    k = csv("kan_results.csv")
    l = e[e["test"] == "uah lodo"].set_index("held_out")
    kk = k[k["data"] == "uah lodo"].set_index("held_out")
    drv = list(l.index)
    cd = CategoryChartData()
    cd.categories = drv
    cd.add_series("reference", [round(v, 3) for v in l["auc_reference"]])
    cd.add_series("extended", [round(v, 3) for v in l["auc_extended"]])
    cd.add_series("KAN", [round(v, 3) for v in kk.loc[drv, "kan"]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.RADAR_MARKERS, Inches(MX), Inches(1.75), Inches(7.6), Inches(5.3), cd)
    ch = gf.chart
    style_chart(ch, size=13, legend=False)
    for ser, c in zip(ch.plots[0].series, (GREY, GOLD, MAROON)):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(2.75)
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(c)
        ser.smooth = False
    ch.value_axis.minimum_scale, ch.value_axis.maximum_scale, ch.value_axis.major_unit = 0.7, 1.0, 0.1
    from pptx.enum.chart import XL_TICK_LABEL_POSITION
    ch.value_axis.tick_label_position = XL_TICK_LABEL_POSITION.NONE
    for i, (nm, v, c) in enumerate((("reference", l["auc_reference"].mean(), GREY), ("extended", l["auc_extended"].mean(), GOLD),
                                    ("KAN", kk["kan"].mean(), MAROON))):
        y = 2.0 + i * 1.45
        tb(s, 8.9, y - 0.2, 3.8, 1.2, [[(f"{v:.3f}", {**NUM, "size": 36, "color": c})], [(f"{nm}, mean AUC", {"size": 14})]])
    bullets(s, 8.9, 5.85, 3.9, ["one spoke per driver, outer = better", "KAN best on all six drivers"], size=14, step=0.45)
    notes(s, "Each spoke is one UAH driver, tested after training on the other five (further out is better). The KAN is the "
             f"outer line on all six drivers. Means: reference {l['auc_reference'].mean():.3f}, extended formula "
             f"{l['auc_extended'].mean():.3f}, KAN {kk['kan'].mean():.3f}. D1, whose camera rarely saw the car ahead, gains the "
             "most from the new signals, which do not depend on the camera.")
    return s


@slide
def s_kan(prs, crest, logo, n):
    s = new(prs)
    title(s, "Learned formula", "Shapes the KAN learned from the data", n, crest)
    s.shapes.add_picture(os.path.join(FIG_DIR, "kan_curves.png"), Inches(MX), Inches(1.5), height=Inches(5.4))
    eq = j("kan_formula.json")["equations"]
    sup = lambda t: t.replace("^2", "²").replace("^3", "³")  # noqa: E731
    cards = [("acceleration", sup(eq["uah"]["accel"]["shape"]), "straight, not squared"),
             ("speed", sup(eq["uah"]["speed"]["shape"]), "curved, matters at high speed"),
             ("gap", sup(eq["uah"]["gap (m)"]["shape"]), "squared, as in the formula")]
    for i, (nm, shape, sub) in enumerate(cards):
        y = 1.7 + i * 1.7
        rect(s, 8.55, y, 4.2, 1.45, MAROON if i < 2 else TINT)
        tb(s, 8.8, y + 0.15, 3.8, 0.4, nm, size=14, color=WHITE if i < 2 else MAROON, bold=True)
        tb(s, 8.8, y + 0.5, 3.8, 0.9, [[(shape, {**NUM, "size": 30, "color": WHITE if i < 2 else MAROON})],
                                       [(sub, {"size": 12, "color": "E8C9CE" if i < 2 else INK})]])
    notes(s, "The KAN learns one curve per input instead of a fixed weight; each curve is matched to a simple formula. On real "
             "UAH drivers acceleration comes out as a straight line (square root on SUMO), not squared: the data does not support "
             "squaring acceleration. Speed comes out curved (cubed on UAH, squared on SUMO). The gap comes out squared. Headway "
             "is U shaped with the calmest point near 1.5 s. The TTC curve rests on very few points. This is the equation learned "
             "from data that Dr. Kurdahi described: the network stays readable.")
    return s


@slide
def s_training(prs, crest, logo, n):
    s = new(prs)
    title(s, "Training", "Accuracy against training length", n, crest)
    cd = CategoryChartData()
    p = csv("learning_curve_ppo_small6_wait5.csv")
    q = csv("learning_curve_dqn_full.csv")
    steps = sorted(set(p["steps"]) | set(q["steps"]))
    cd.categories = [f"{x / 1e6:g}M" for x in steps]
    cd.add_series("PPO", [round(100 * float(p.set_index("steps")["balanced_acc"].get(x, np.nan)), 1) for x in steps])
    cd.add_series("DQN", [round(100 * float(q.set_index("steps")["balanced_acc"].get(x, np.nan)), 1) for x in steps])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(MX), Inches(1.5), Inches(8.9), Inches(5.3), cd)
    ch = gf.chart
    style_chart(ch, size=12)
    for ser, c in zip(ch.plots[0].series, (MAROON, BLUE)):
        ser.format.line.color.rgb = rgb(c)
        ser.format.line.width = Pt(3)
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(c)
        ser.marker.format.line.color.rgb = rgb(c)
        ser.smooth = False
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 30, 80, 10
    va.major_gridlines.format.line.color.rgb = rgb(LINE)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = '0"%"', False
    ch.category_axis.tick_labels.font.size = Pt(10)
    tb(s, 9.9, 2.0, 2.9, 1.4, [[("2.6 min", {**NUM, "size": 40, "color": MAROON})], [("per million steps", {"size": 14})]])
    bullets(s, 9.9, 3.5, 3.0, ["1 step = 1 s of one driver", "PPO flat at 71 to 74%", "DQN catches up after exploring",
                               "limit is information, not time"], size=14, step=0.6)
    notes(s, "One step is one second of one driver and one button press; a million steps is about 30,000 drivers and takes 2.6 "
             "minutes on the laptop, because the drivers are pre recorded tables and the network is small. PPO is at about 72% "
             "after half a million steps and stays between 71 and 74% up to 10 million. DQN explores at random for its first 3 "
             "million steps, then reaches the same level. Training longer does not help: the limit is the information.")
    return s


@slide
def s_waterfall(prs, crest, logo, n):
    s = new(prs)
    title(s, "Agents", "Only new information moves the ceiling", n, crest)
    a = csv("ablation_state_reward.csv").set_index("config")
    f = csv("fsm_vs_joint.csv")
    f = f[f["environment"] == "all"].set_index("agent")
    base = 100 * a.loc["small6_wait5", "balanced_acc"]
    steps = [("previous best\n8 inputs", base, None), ("+ new signals", 100 * f.loc["joint_plain", "balanced_acc"], MAROON),
             ("+ risk reward", 100 * f.loc["joint_risk", "balanced_acc"], GOLD),
             ("+ 3x training", 100 * f.loc["joint_risk_long", "balanced_acc"], GOLD)]
    lo, hi = 65.0, 82.0
    x0, y0, H, bw, gap = MX + 0.3, 6.3, 4.2, 1.65, 0.6
    yof = lambda v: y0 - (v - lo) / (hi - lo) * H  # noqa: E731
    groups, prev = [], None
    for i, (nm, v, col) in enumerate(steps):
        x = x0 + i * (bw + gap)
        if col is None:
            r = rect(s, x, yof(v), bw, y0 - yof(v), GREY, shape=MSO_SHAPE.RECTANGLE)
            lab = tb(s, x, yof(v) - 0.55, bw, 0.5, f"{v:.1f}%", size=20, bold=True, align=PP_ALIGN.CENTER)
        else:
            r = rect(s, x, yof(v), bw, max(yof(prev) - yof(v), 0.05), col, shape=MSO_SHAPE.RECTANGLE)
            lab = tb(s, x, yof(v) - 0.55, bw, 0.5, f"+{v - prev:.1f}", size=20, bold=True, align=PP_ALIGN.CENTER, color=col)
        cap = tb(s, x, y0 + 0.1, bw, 0.6, nm, size=13, align=PP_ALIGN.CENTER, color=INK)
        grp = [r.shape_id, lab.shape_id, cap.shape_id]
        if prev is not None:
            ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x - gap), Inches(yof(prev)), Inches(x), Inches(yof(prev)))
            ln.line.color.rgb, ln.line.width = rgb(GREY), Pt(1)
            ln.line.dash_style = 4
            grp.append(ln.shape_id)
        groups.append(grp)
        prev = v
    end = steps[-1][1]
    bullets(s, 10.3, 4.75, 2.6, ["new signals: +7.3", "reward, longer: +0.4"], size=14, step=0.5)
    tb(s, 10.3, 3.2, 2.5, 1.6, [[(f"{end:.1f}%", {**NUM, "size": 44, "color": MAROON})], [("SUMO test drivers", {"size": 14})]])
    notes(s, f"Waterfall of the agent's accuracy on the simulated test drivers. Previous best {base:.1f}%. Adding the new "
             f"signals to its state (physics residual, speed vs limit and traffic, danger) adds {steps[1][1] - base:.1f} points. "
             f"The risk weighted reward adds {steps[2][1] - steps[1][1]:.1f}, three times longer training "
             f"{steps[3][1] - steps[2][1]:.1f}. New information moves the ceiling; the reward changes how the agent spends what "
             "it has (dangerous drivers labelled in 19 s instead of 22 s, on 58 such drivers). Each step appears on click.")
    return s, groups


@slide
def s_fsm(prs, crest, logo, n):
    s = new(prs)
    title(s, "Scenario experts", "Three experts switched by a state machine", n, crest)
    cx, cy = 3.6, 4.55
    states = {"highway": (cx - 2.4, cy - 1.6), "urban": (cx + 2.4, cy - 1.6), "weather": (cx, cy + 1.3)}
    groups = []
    road = labelled(s, cx - 1.1, 1.6, 2.2, 0.65, "road type", fill=TINT, color=MAROON, size=15)
    groups.append([road.shape_id])
    shapes = {}
    for nm, (x, y) in states.items():
        c = labelled(s, x - 0.85, y - 0.6, 1.7, 1.2, f"{nm}\nexpert", fill=MAROON, size=16, shape=MSO_SHAPE.OVAL)
        shapes[nm] = c
    grp = [c.shape_id for c in shapes.values()]
    for a_, b_ in (("highway", "urban"), ("urban", "weather"), ("weather", "highway")):
        (x1, y1), (x2, y2) = states[a_], states[b_]
        dx, dy = x2 - x1, y2 - y1
        d = (dx ** 2 + dy ** 2) ** 0.5
        ar = arrow(s, x1 + dx / d * 0.95, y1 + dy / d * 0.7, x2 - dx / d * 0.95, y2 - dy / d * 0.7, color=GOLD)
        grp.append(ar.shape_id)
    for nm, (x, y) in states.items():
        ar = arrow(s, cx + (x - cx) * 0.25, 2.27, x + (cx - x) * 0.25, y - 0.62, color=GREY)
        grp.append(ar.shape_id)
    groups.append(grp)
    f = csv("fsm_vs_joint.csv")
    rows = [["road type", "one joint agent", "3 experts + FSM"]]
    fill = {}
    for i, e in enumerate(("highway", "urban", "weather", "all"), 1):
        jv = f[(f["agent"] == "joint_risk") & (f["environment"] == e)]["balanced_acc"].iloc[0]
        ev = f[(f["agent"] == "fsm_experts") & (f["environment"] == e)]["balanced_acc"].iloc[0]
        rows.append([e, f"{jv:.1%}", f"{ev:.1%}"])
        fill[(i, 1 if jv > ev else 2)] = "EBD3A8"
    t = table(s, 7.3, 2.0, 5.45, 3.0, rows, col_w=[1.6, 1.9, 1.95], size=16, cell_fill=lambda i, k: fill.get((i, k)))
    bullets(s, 7.3, 5.25, 5.45, ["experts win on urban and weather", "same rule, thresholds shifted per road",
                                 "fusion: one agent + road threshold"], size=14, step=0.45)
    notes(s, "Three experts, one per road type, each trained and tested only on its own road type. The state machine reads the "
             "road type and hands the car to that expert (the gold arrows are the switches when the road type changes). Overall "
             "the two approaches tie; the experts win on urban roads and in weather, the joint agent on the highway where most "
             "data is. All experts use the same rule (conservative lowest score, aggressive highest) with thresholds shifted per "
             "road type: highway 17 / 21 / 26, urban 12 / 16 / 23, weather 14 / 19 / 22. So the fusion is one shared agent "
             "with a road type specific threshold. The machine builds on click.")
    return s, groups


@slide
def s_autobahn(prs, crest, logo, n):
    s = new(prs)
    title(s, "Germany", "Simulated German autobahn", n, crest)
    s.shapes.add_picture(os.path.join(FIG_DIR, "autobahn.gif"), Inches(MX), Inches(1.5), Inches(SW - 2 * MX))
    d = pd.read_csv(os.path.join(DATA_DIR, "clean", "planted_de.csv.gz"), usecols=["vid", "label"]).drop_duplicates("vid")
    mix = d["label"].value_counts()
    cd = CategoryChartData()
    order = ["normal", "conservative", "aggressive", "truck"]
    cd.categories = order
    cd.add_series("vehicles", [int(mix.get(k, 0)) for k in order])
    gf = s.shapes.add_chart(XL_CHART_TYPE.PIE, Inches(MX), Inches(4.65), Inches(3.4), Inches(2.6), cd)
    ch = gf.chart
    style_chart(ch, size=10, legend=False)
    pl = ch.plots[0]
    for i, c in enumerate((GOLD, BLUE, MAROON, GREY)):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    pl.has_data_labels = True
    pl.data_labels.show_category_name, pl.data_labels.show_percentage, pl.data_labels.show_value = True, True, False
    pl.data_labels.font.size = Pt(9)
    table(s, 4.6, 4.85, SW - MX - 4.6, 1.9, [["", "120 km/h section", "no-limit section"],
                                             ["lanes", "3", "3"], ["keep right / overtake on right", "all / aggressive only", "all / aggressive only"]],
          col_w=[3.1, 2.6, 2.433], size=13)
    notes(s, "A German style motorway added to the simulation (animation: 30 s of run 0, cars drawn longer to be visible). "
             "2 km at 120 km/h, then 2 km without a speed limit, where each type drives its own preferred top speed; a stronger "
             "keep right rule; overtaking on the right only by aggressive drivers; 15% trucks limited to 80 km/h in the right "
             "lane (pie: the traffic mix). Medians on the unlimited section: aggressive 128, normal 118, conservative 110, "
             "trucks 79 km/h. 10 runs of about 650 vehicles.")
    return s


@slide
def s_german(prs, crest, logo, n):
    s = new(prs)
    title(s, "Germany", "Transfer to the autobahn", n, crest)
    g = csv("german_transfer.csv")
    a = g[g["test"] == "autobahn test"].set_index("agent")
    rows = [["agent", "trained on", "inputs", "accuracy", "decides after"]]
    spec = [("A_naive", "other roads", "previous 8"), ("B_relative", "other roads", "+ relative speeds, physics"),
            ("C_de_onehot", "other + German", "B + country one-hot"), ("D_de_only", "German only", "as B")]
    fill = {}
    for i, (k, tr, inp) in enumerate(spec, 1):
        rows.append([k[0], tr, inp, f"{a.loc[k, 'balanced_acc']:.1%}", f"{a.loc[k, 'seconds']:.0f} s"])
        fill[(i, 3)] = heat(a.loc[k, "balanced_acc"], 0.68, 0.82)
    best = a["balanced_acc"].idxmax()
    table(s, MX, 1.8, 8.6, 3.6, rows, col_w=[0.9, 2.1, 3.0, 1.3, 1.3], size=15, cell_fill=lambda i, k: fill.get((i, k)))
    rect(s, 9.6, 1.8, 3.13, 3.6, MAROON)
    tb(s, 9.85, 2.1, 2.7, 3.1, [[(f"{a.loc[best, 'balanced_acc']:.1%}", {**NUM, "size": 46, "color": WHITE})],
                                [("never saw a German road", {"size": 15, "color": WHITE, "bold": True})],
                                [("relative speeds and the physics residual carry over between countries",
                                  {"size": 13, "color": "E8C9CE"})]])
    bullets(s, MX, 5.65, 8.6, ["tested on German runs never seen in training", "C and D: small German set, single runs"],
            size=14, step=0.45)
    notes(s, "Agent A: trained on the other simulated roads with the old inputs. Agent B: same roads, plus speed relative to the "
             "limit and to the traffic and the physics residual. B never saw a German road and is the best on the autobahn: "
             "relative features carry over between countries, raw speed does not. Adding German data and a country one hot (C), "
             "or German data only (D), was worse here: the German training set is six times smaller and both decide much "
             "earlier. Single runs; C should be repeated with more seeds.")
    return s


@slide
def s_highd(prs, crest, logo, n):
    s = new(prs)
    title(s, "Germany", "Real highD, rule-violation labels", n, crest)
    h = csv("german_transfer_highd_proxy.csv").iloc[0]
    cd = CategoryChartData()
    cd.categories = ["tailgating or speeding", "neither, while following"]
    cd.add_series("windows", [round(100 * h["share_proxy_aggressive"], 1), round(100 * (1 - h["share_proxy_aggressive"]), 1)])
    gf = s.shapes.add_chart(XL_CHART_TYPE.DOUGHNUT, Inches(MX), Inches(1.6), Inches(5.0), Inches(4.8), cd)
    ch = gf.chart
    style_chart(ch)
    pl = ch.plots[0]
    for i, c in enumerate((MAROON, GREY)):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0"%"', False
    pl.data_labels.font.size, pl.data_labels.font.color.rgb = Pt(14), rgb(WHITE)
    tb(s, MX + 2.5 - 1.0, 3.55, 2.0, 0.9, [[(f"{h['share_proxy_aggressive']:.0%}", {**NUM, "size": 34, "color": MAROON})],
                                          [("break a rule", {"size": 11, "color": MUTED})]], align=PP_ALIGN.CENTER)
    rows = [["AUC", "all terms", "without gap and limit"],
            ["reference", f"{h['auc_reference_all_terms']:.2f}", f"{h['auc_reference_without_gap_and_limit']:.2f}"],
            ["extended", f"{h['auc_extended_all_terms']:.2f}", f"{h['auc_extended_without_gap_and_limit']:.2f}"]]
    table(s, 6.3, 2.2, 6.43, 2.4, rows, col_w=[1.9, 2.0, 2.53], size=17,
          cell_fill=lambda i, k: ("EBD3A8" if (i, k) == (2, 2) else (LINE if k == 1 and i else None)))
    bullets(s, 6.3, 4.85, 6.43, ["rule: under 0.9 s headway or 20% over the limit", "grey column is circular (rules use gap, limit)",
                                 "extended still finds violators: 0.83"], size=14, step=0.5)
    notes(s, f"highD has no labels, so rule violations serve as rough labels ({int(h['windows']):,} windows; the doughnut shows "
             "the share flagged). With all terms both formulas look excellent, but that is circular: the rules are defined by "
             "the gap and the speed limit. Without those two terms the extended formula still finds the violators (0.83) and "
             "the reference does not (0.62); most of it comes from the physics residual, which partly overlaps with close "
             "following.")
    return s


@slide
def s_replay(prs, crest, logo, n):
    s = new(prs)
    title(s, "Agents", "Replay: how the agent decides", n, crest)
    w = (SW - 2 * MX - 0.4) / 2
    h = w * 607 / 1080
    for i, (f, cap) in enumerate((("explainer_correct_follows.gif", "follows a car · labelled normal · correct"),
                                  ("explainer_wrong_never_follows.gif", "never close to a car · labelled conservative · wrong"))):
        x = MX + i * (w + 0.4)
        s.shapes.add_picture(os.path.join(GIFS, f), Inches(x), Inches(1.75), Inches(w))
        circle_num(s, x - 0.15, 1.6, i + 1, d=0.45)
        bullets(s, x, 1.9 + h, w, [cap], size=14)
    notes(s, "Replay of two normal simulated drivers as the previous agent (8 inputs) saw them, second by second. When the car "
             "follows another car the following clue appears and the agent is right; when it never does, the agent falls back "
             "on conservative. The new signals give it information in exactly that second case.")
    return s


@slide
def s_rag(prs, crest, logo, n):
    s = new(prs)
    title(s, "Context", "RAG: where it plugs in", n, crest)
    y, w, gap = 2.6, 2.45, 0.55
    items = [("window", "+ situation", TINT, MAROON), ("retrieve", "similar normal\nwindows", MAROON, WHITE),
             ("context", "percentile", MAROON, WHITE), ("agent", "state", DEEP, WHITE)]
    groups = []
    for i, (t, sub, fill, col) in enumerate(items):
        x = MX + 0.3 + i * (w + gap)
        b = labelled(s, x, y, w, 1.4, f"{t}\n{sub}", fill=fill, color=col, size=18)
        grp = [b.shape_id]
        if i:
            grp.append(arrow(s, x - gap + 0.04, y + 0.7, x - 0.04, y + 0.7).shape_id)
        groups.append(grp)
    k = labelled(s, MX + 0.3 + (w + gap), 4.7, w, 1.1, "road type · traffic state\nspeed limit · country", fill=WHITE,
                 color=MAROON, size=14)
    k.line.color.rgb = rgb(MAROON)
    groups.append([k.shape_id, arrow(s, MX + 0.3 + (w + gap) + w / 2, 4.68, MX + 0.3 + (w + gap) + w / 2, y + 1.42).shape_id])
    bullets(s, MX + 0.1, 6.15, 6.0, ["built in October, not in the models yet", "next: context percentile in the agent state"],
            size=14, step=0.45)
    kn = labelled(s, MX + 0.3 + 2 * (w + gap), 4.7, 2 * w + gap, 1.1, "knowledge files (rules per country)\n→ explanation of each decision",
                  fill=TINT, color=INK, size=14, bold=False)
    groups.append([kn.shape_id])
    notes(s, "Where the RAG stands: built in early October (model/rag.py) as a separate experiment; it scores a window against "
             "normal windows retrieved from the same road type and traffic state. It helped only in mixed simulated traffic and "
             "lost a little on UAH, so it is not part of the formula or the agents today. Plan: put the retrieved context "
             "percentile into the agent's state (how unusual this driving is for this road, traffic and country), the natural "
             "way to carry the model to German roads, and use the knowledge files only to explain decisions. Builds on click.")
    return s, groups


@slide
def s_next(prs, crest, logo, n):
    s = new(prs, dark=True)
    tb(s, MX, 0.6, 10, 0.4, "NEXT STEPS", size=13, color="E8C9CE", bold=True)
    tb(s, MX, 1.0, 11, 0.8, [[("What comes next", {"font": HEAD, "size": 34, "color": WHITE})]])
    items = ["smooth the label: probability, band, hysteresis", "acceleration straight, speed curved (KAN)",
             "context percentile (RAG) in the agent state", "German transfer: more seeds, each rule on highD",
             "full DriveDNA once access is granted", "intervention (throttle damping) from the components"]
    w = (SW - 2 * MX - 0.6) / 3
    for i, t in enumerate(items):
        x, y = MX + (i % 3) * (w + 0.3), 2.4 + (i // 3) * 2.2
        rect(s, x, y, w, 1.85, "5E1724")
        circle_num(s, x + 0.25, y + 0.25, i + 1, d=0.55, fill=GOLD)
        tb(s, x + 0.25, y + 0.95, w - 0.5, 0.85, t, size=15, color=WHITE)
    notes(s, "Next: replace the hard cut-off with a probability, a borderline band and hysteresis; adopt the shapes the KAN learned (acceleration straight, "
             "speed curved) after discussing them; put the RAG context percentile into the agent's state; repeat the German "
             "transfer with more seeds and check each rule on highD separately; run full DriveDNA once the owners grant access; "
             "and from Dr. Kurdahi's meeting, use the separate components of the index to decide when a system should intervene, "
             "for example damping the throttle.")
    return s


def set_simple_notes(prs):
    """speaker notes from scripts/speaker_notes_long.py, one entry per slide, in slide order"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from speaker_notes_long import NOTES
    assert len(NOTES) == len(prs.slides), (len(NOTES), len(prs.slides))
    for s, n in zip(prs.slides, NOTES):
        notes(s, "SAY: " + n["say"] + ("\n\nIF ASKED: " + n["asked"] if n.get("asked") else ""))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prs = Presentation(SRC)
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    with zipfile.ZipFile(SRC) as z:
        tmp = tempfile.mkdtemp()
        crest, logo = os.path.join(tmp, "crest.png"), os.path.join(tmp, "logo.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
        open(logo, "wb").write(z.read("ppt/media/image-1-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    TOTAL[0] = len(SLIDES)
    for i, f in enumerate(SLIDES):
        r = f(prs, crest, logo, i + 1)
        s, groups = r if isinstance(r, tuple) else (r, None)
        if groups:
            add_builds(s, groups)
        add_transition(s)
    set_simple_notes(prs)
    prs.save(OUT)
    print("written", OUT, len(SLIDES), "slides")


if __name__ == "__main__":
    main()
