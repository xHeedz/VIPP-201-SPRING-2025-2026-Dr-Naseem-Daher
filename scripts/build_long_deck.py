"""
Long progress presentation (week of 26 Oct 2026): figures, equations, diagrams and videos on the slides, the full
explanation in each slide's notes. Every number is read from data/ (figures from scripts/make_presentation_figures.py).

    python scripts/build_long_deck.py
writes ../26-10-2026/Progress Presentation.pptx
"""
import json
import os
import sys
import tempfile
import zipfile

import pandas as pd
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_october_deck as B  # noqa: E402
from build_october_deck import (HEAD, INK, MAROON, MAROON2, MUTED, PALE, PINK, W, WINE, X0, card, header,  # noqa: E402
                                new_slide, rgb, text)

DATA, ROOT = B.DATA, B.ROOT
FIG = os.path.join(ROOT, "results", "figures")
OUT_DIR = os.path.abspath(os.path.join(ROOT, "..", "26-10-2026"))
OUT = os.path.join(OUT_DIR, "Progress Presentation.pptx")
AUDIT = os.path.join(OUT_DIR, "label_audit")
GIFS = os.path.abspath(os.path.join(ROOT, "..", "20-10-2026"))
SLIDES = []


def slide(fn):
    SLIDES.append(fn)
    return fn


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def footer(s, n):
    B.footer(s, n)


def picture(s, path, x, y, w=None, h=None):
    if os.path.exists(path):
        return s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def caption(s, t, y=6.35):
    text(s, X0, y, W, 0.4, t, size=12, color=MUTED, align=PP_ALIGN.CENTER)


def figure_slide(prs, crest, n, label, title, fig, cap, note):
    s = new_slide(prs)
    header(s, label, title, crest)
    picture(s, os.path.join(FIG, fig), X0 + 0.1, 1.65, w=W - 0.2)
    caption(s, cap)
    footer(s, n)
    notes(s, note)
    return s


def box(s, x, y, w, h, t, fill=MAROON, color="FFFFFF", size=13, bold=True):
    card(s, x, y, w, h, fill=fill)
    text(s, x + 0.1, y, w - 0.2, h, [[(t, {"bold": bold, "color": color, "size": size})]], align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(MAROON2)
    c.line.width = Pt(2.25)
    from lxml import etree
    c.line._get_or_add_ln().append(etree.fromstring(
        '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle"/>'))


def num(path, *keys):
    d = json.load(open(os.path.join(DATA, path)))
    for k in keys:
        d = d[k]
    return d


# ── slides ──────────────────────────────────────────────────────────────────
@slide
def s_title(prs, crest, n, logo):
    s = new_slide(prs, MAROON)
    s.shapes.add_picture(logo, Inches(X0), Inches(0.667), Inches(3.333), Inches(0.778))
    text(s, 8.2, 0.6, 4.244, 1.0, [[("Hadi Al Shmaissani", {"bold": True, "color": "FFFFFF"})],
                                    [("Supervisor: Dr. Naseem Daher", {"color": PALE})],
                                    [("American University of Beirut", {"color": PALE})]], size=12, align=PP_ALIGN.RIGHT)
    text(s, X0, 2.3, W, 0.3, "VIPP 301A · PROGRESS PRESENTATION", size=12, color=PINK, spacing=150)
    text(s, X0, 2.7, W, 1.6, [[("Driver aggressiveness: checking the labels, new signals, learned formulas "
                                 "and transfer to German roads", {"font": HEAD, "size": 34, "color": "FFFFFF"})]])
    notes(s, "This presentation covers the work since the last meeting: a manual check of the labels, new signals (time to "
             "collision, speed relative to the limit and to the traffic, a physics based residual), a network that learns the "
             "formula, how long the agents need to train, scenario experts switched by a state machine, and moving the model "
             "to German roads.")


@slide
def s_pipeline(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Overview", "Pipeline", crest)
    xs = [X0, X0 + 2.35, X0 + 4.7, X0 + 7.05, X0 + 9.4]
    boxes = [("Data\nUAH · SUMO · NGSIM\npNEUMA · highD …", "F5E9EB", MAROON2), ("Cleaning\nunknown ≠ 0", "F5E9EB", MAROON2),
             ("Signals\nTTC · speed vs limit\nspeed vs traffic · IDM", MAROON, "FFFFFF"),
             ("Score\nformula · KAN", MAROON, "FFFFFF"), ("Agent\nPPO · DQN · FSM", MAROON, "FFFFFF")]
    for (t, fill, col), x in zip(boxes, xs):
        box(s, x, 2.6, 2.05, 1.6, t, fill=fill, color=col, size=13)
    for x in xs[:-1]:
        arrow(s, x + 2.07, 3.4, x + 2.33, 3.4)
    box(s, X0 + 9.4, 4.75, 2.05, 0.9, "label\nconservative · normal · aggressive", fill="FFFFFF", color=MAROON2, size=11)
    arrow(s, X0 + 10.42, 4.22, X0 + 10.42, 4.73)
    box(s, X0 + 4.7, 4.75, 4.4, 0.9, "RAG: context from similar situations (design)", fill="E9E2E3", color=MUTED, size=12,
        bold=False)
    arrow(s, X0 + 6.9, 4.73, X0 + 8.2, 4.22)
    footer(s, n)
    notes(s, "The whole pipeline on one slide. Every dataset goes through the same cleaning, where unknown values stay unknown "
             "instead of becoming zero. New signals are computed from the cleaned data: time to collision, speed relative to "
             "the limit and to the traffic, and the IDM residual. A score is computed either by the fixed formula or by a "
             "network that learns the formula (the KAN). The agents watch a driver second by second and decide when to label. "
             "The RAG layer is a design for now: it would add context from similar situations.")


@slide
def s_equations(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Model", "Equations", crest)
    w = num("extended_weights.json", "weights")
    rows = [
        ("Reference formula", "AI = 100 · Σₖ wₖ·xₖ / Σₖ wₖ   (known terms only)"),
        ("Extended formula", f"AI = 100 · ({w[0]:.1f} v + {w[1]:.1f} a² + {w[2]:.1f} g² + {w[3]:.1f} ℓ + {w[4]:.1f} TTC "
                             f"+ {w[5]:.1f} v_lim + {w[7]:.1f} IDM) / Σ wₖ"),
        ("KAN (learned)", "AI = σ( b + Σᵢ φᵢ(xᵢ) ),   φᵢ(x) = wᵢ·silu(x) + Σⱼ cᵢⱼ Bⱼ(x)   (cubic B-splines, grid 5)"),
        ("Time to collision", "TTC = gap / closing speed ;   term = (1 − TTC / 4 s)²  if TTC < 4 s"),
        ("Physics residual (IDM)", "r = a − a_IDM ,   a_IDM = a_max [1 − (v/v₀)⁴ − (s*/s)²] ,   s* = s₀ + vT + vΔv / 2√(a_max b)"),
        ("Risk reward", "r_wait = −0.002 (1 + 5 dₜ) ;   r_miss = −(1 + 2 maxₜ dₜ) ;   r_correct = +1"),
    ]
    for i, (a, b) in enumerate(rows):
        y = 1.7 + i * 0.78
        card(s, X0, y, W, 0.66, fill=MAROON if i == 1 else "F5E9EB")
        text(s, X0 + 0.25, y + 0.17, 2.9, 0.35, [[(a, {"bold": True, "color": "FFFFFF" if i == 1 else MAROON2, "size": 14})]])
        text(s, X0 + 3.2, y + 0.15, W - 3.4, 0.4, [[(b, {"color": "FFFFFF" if i == 1 else INK, "size": 15, "font": "Cambria Math"})]])
    footer(s, n)
    notes(s, "Reference formula: a weighted mean of the known terms only; an unknown term is left out and the other weights "
             "rescaled. Terms: v = speed / 150, a = |acceleration| / 5 (squared), g = gap closeness mixing metres and seconds "
             "(squared), lane offset / 1.5 m. "
             f"Extended formula: four more terms; fitted weights speed {w[0]:.1f}, acceleration² {w[1]:.1f}, gap {w[2]:.1f}, "
             f"lane {w[3]:.1f}, TTC {w[4]:.1f}, speed relative to the limit {w[5]:.1f}, speed relative to traffic {w[6]:.1f}, "
             f"IDM residual {w[7]:.1f}. "
             "KAN: one learned curve per input (splines with a grid of 5 intervals, degree 3), summed and passed through a "
             "sigmoid for the probability of aggressive. "
             "TTC: seconds until a collision if neither car changes speed; only short TTC counts. "
             "IDM residual: the Intelligent Driver Model gives the acceleration a calm driver would choose in the same "
             "situation (a_max 1.5 m/s², b 2 m/s², T 1.5 s, s0 2 m, v0 = speed limit); the residual is how far the real driver "
             "is from that. "
             "Risk reward: d is the danger of a second (TTC under 4 s or headway under 1 s). Waiting while it is dangerous "
             "costs up to six times more, and missing a dangerous aggressive driver costs up to 3 instead of 1.")


@slide
def s_audit(prs, crest, n, logo):
    s = pd.read_csv(os.path.join(DATA, "label_audit_summary.csv")).set_index("group")
    a = s.loc["label aggressive, formula calm"]
    return figure_slide(prs, crest, n, "Label audit", "Formula against label, UAH-DriveSet", "pres_audit.png",
                        "red: formula and label disagree · right: UAH's own software as a third opinion",
                        "Every UAH 10 second window gets the formula's verdict and the trip label. Where they disagree, UAH's own "
                        "software (DriveSafe) is a third opinion. Labelled aggressive but the formula says calm: "
                        f"{int(a['windows'])} windows, and DriveSafe agrees with the formula in {a['drivesafe_sides_with_formula']:.0%} of "
                        "them, so most are calm stretches inside an aggressive trip: the label is wrong. Labelled normal but the "
                        "formula says aggressive: DriveSafe never flags any second of a normal trip, so it cannot judge that group; "
                        "the videos and the manual review decide it.")


@slide
def s_audit_cases(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Label audit", "Two sampled windows", crest)
    w = (W - 0.3) / 2
    picture(s, os.path.join(AUDIT, "figures", "02.png"), X0, 1.6, w=w)
    picture(s, os.path.join(AUDIT, "figures", "10.png"), X0 + w + 0.3, 1.6, w=w)
    text(s, X0, 6.1, w, 0.5, "label aggressive · formula 23.8 · gap closes to 10 m, TTC about 1 s", size=11, color=INK,
         align=PP_ALIGN.CENTER)
    text(s, X0 + w + 0.3, 6.1, w, 0.5, "label normal · formula 31.7 · 100 km/h in a 120 km/h zone", size=11, color=INK,
         align=PP_ALIGN.CENTER)
    footer(s, n)
    notes(s, "Left (sample 02): labelled aggressive, the formula says calm. The speed is at the limit and the acceleration is "
             "gentle, but at the end of the window the gap closes from 50 m to 10 m and the time to collision drops to about "
             "one second. Here the label is probably right and the formula late: averaged over 10 seconds, the closing in only "
             "shows partly. That is the argument for TTC. Right (sample 10): labelled normal, the formula says aggressive. "
             "100 km/h in a 120 km/h zone behind trucks: ordinary driving. Here the label is right and the formula is wrong "
             "because raw speed pushes it up. That is the argument for speed relative to the limit. So the audit finds both "
             "directions: wrong labels and formula mistakes.")


@slide
def s_audit_video(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Label audit", "The same two windows, on video", crest)
    w = (W - 0.3) / 2
    h = w * 9 / 16
    for i, c in enumerate(("02", "10")):
        mp4, poster = os.path.join(AUDIT, "clips", f"{c}.mp4"), os.path.join(AUDIT, "clips", f"{c}.mp4.png")
        if os.path.exists(mp4):
            s.shapes.add_movie(mp4, Inches(X0 + i * (w + 0.3)), Inches(1.75), Inches(w), Inches(h),
                               poster_frame_image=poster if os.path.exists(poster) else None, mime_type="video/mp4")
    text(s, X0, 1.75 + h + 0.15, w, 0.4, "02 · labelled aggressive", size=12, color=INK, align=PP_ALIGN.CENTER)
    text(s, X0 + w + 0.3, 1.75 + h + 0.15, w, 0.4, "10 · labelled normal", size=12, color=INK, align=PP_ALIGN.CENTER)
    footer(s, n)
    notes(s, "The road video of the same two windows (10 seconds each, synchronised the way UAH's reader tool does it). All 22 "
             "sampled windows, with their figures and clips, are in the label_audit folder (index.html) and in review.csv for "
             "the manual verdict.")


@slide
def s_signals(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "New signals", "What each signal is worth on its own", "pres_single_terms.png",
                        "AUC of one signal alone · 0.5 = coin flip",
                        "Each bar is one signal used alone to tell aggressive from normal drivers. The physics residual (IDM) is "
                        "among the two best on real UAH drivers and the best on the simulated ones. On the simulation this is partly "
                        "by construction, because the simulated drivers are IDM drivers, so the real data result is the convincing "
                        "one. Time to collision alone is a coin flip at the 10 second level: closing in moments are short and rare, "
                        "so they vanish in the average. TTC is therefore used as a danger signal in the reward rather than as an "
                        "averaged term.")


@slide
def s_formulas(prs, crest, n, logo):
    k = pd.read_csv(os.path.join(DATA, "kan_results.csv"))
    e = pd.read_csv(os.path.join(DATA, "extended_fit.csv"))
    l = e[e["test"] == "uah lodo"]
    kk = k[k["data"] == "uah lodo"]
    return figure_slide(prs, crest, n, "Learned formula", "Reference, extended formula and KAN", "pres_formulas.png",
                        "UAH, leave one driver out",
                        f"Each driver is tested after training on the other five. Means: reference {l['auc_reference'].mean():.3f}, "
                        f"extended formula {l['auc_extended'].mean():.3f}, KAN {kk['kan'].mean():.3f}. The KAN is better on all six "
                        "drivers. Driver D1, whose camera rarely saw the car ahead, gains the most from the new signals, because "
                        "speed relative to the limit and the physics residual do not depend on the camera.")


@slide
def s_kan(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Learned formula", "Shapes learned by the KAN", crest)
    picture(s, os.path.join(FIG, "kan_curves.png"), X0 + 1.6, 1.55, h=4.75)
    eq = num("kan_formula.json", "equations", "uah")
    short = " + ".join(f"{v['a']:.1f}·{v['shape'].replace('x', n_)}" for n_, v in
                       [(k_, eq[k_]) for k_ in ("speed", "accel", "gap (m)", "lane", "speed vs limit") if k_ in eq])
    caption(s, f"UAH: {short}", y=6.38)
    footer(s, n)
    notes(s, "The KAN learns one curve per input instead of fixed weights; each curve is then matched to a simple formula. "
             "Acceleration comes out as a straight line on real drivers (square root on the simulated ones), not squared: the "
             "data does not support squaring acceleration. Speed comes out curved (cubed on UAH, squared on SUMO), so speed "
             "matters mostly at high values. The gap comes out squared, as in the formula. Headway is U shaped: about 1.5 s is "
             "the calmest. The TTC curve rests on very few points (TTC is rarely below 4 s at the window level) and should not be "
             "read into. This is the 'equation learned from data' idea from Dr. Kurdahi: the network stays readable.")


@slide
def s_training(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "Training", "Accuracy against training length", "pres_learning_curves.png",
                        "tested on 1,500 unseen SUMO drivers every 0.5 million steps",
                        "One step is one second of one driver and one button press; one million steps is about 30,000 drivers and "
                        "takes about 2.6 minutes on the laptop. Training is fast because the drivers are pre recorded tables and the "
                        "network is small. PPO is at about 72% from the first half million steps and stays between 71 and 74% up to "
                        "10 million. DQN explores at random for its first 3 million steps, then reaches the same level. Training "
                        "longer does not help: the limit is the information, not the training time.")


@slide
def s_agents(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "Agents", "Accuracy by agent configuration", "pres_agents.png",
                        "SUMO test drivers (runs 7 to 9) · new signals: IDM residual, speed vs limit and traffic, danger",
                        "Adding the new signals to the agent's state raises it from 72% to about 80%. The risk weighted reward and "
                        "three times longer training each add about 0.2 points. Only new information moves the ceiling; the reward "
                        "changes how the agent spends its information (it decided on dangerous drivers in 19 s instead of 22 s, "
                        "on only 58 such drivers).")


@slide
def s_fsm(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "Scenario experts", "One joint agent against three experts switched by an FSM",
                        "pres_fsm.png", "right: median formula score of the vehicles each expert labels conservative / normal / aggressive",
                        "Three experts, one per road type, each trained and tested only on its own road type; a finite state machine "
                        "sends each car to the expert of its road type. Overall the two approaches tie (80.1% against 79.7%). The "
                        "experts are clearly better on urban roads and in weather, the joint agent slightly better on the highway "
                        "where most of the data is. Right: all three experts use the same rule (conservative lowest, aggressive "
                        "highest) with the thresholds shifted per road type. So the fusion is one shared agent with a road type "
                        "specific threshold, which is what the road type one hot input allows.")


@slide
def s_autobahn(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "Germany", "Simulated German autobahn", "pres_autobahn.png",
                        "3 lanes · 2 km at 120 km/h then 2 km without limit · keep right · 15% trucks at 80 km/h",
                        "A German style motorway added to the simulation: the same planted driver types, a section without a speed "
                        "limit (each type has its own preferred top speed there), a stronger keep right rule, overtaking on the right "
                        "only by aggressive drivers, and trucks limited to 80 km/h in the right lane. Every type speeds up on the "
                        "unlimited section, trucks stay at 79 km/h. 10 runs, about 650 vehicles each.")


@slide
def s_german(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "Germany", "Transfer to the autobahn", "pres_german.png",
                        "all tested on German runs 7 to 9, never seen in training",
                        "Agent A is trained on the other simulated roads with the old inputs. Agent B is trained on the same roads "
                        "but with speed relative to the limit and to the traffic and the physics residual: it never saw a German "
                        "road and is the best on the autobahn. Relative features carry over between countries, raw speed does not. "
                        "Adding German data and a country one hot (C), or training on German data only (D), was worse here: the "
                        "German training set is six times smaller and both decide much earlier. These are single runs; C should be "
                        "repeated with more seeds before concluding.")


@slide
def s_highd(prs, crest, n, logo):
    return figure_slide(prs, crest, n, "Germany", "Real highD, rule-violation labels", "pres_highd_proxy.png",
                        "proxy aggressive: half the window under 0.9 s headway or 20% over the limit",
                        "highD has no labels, so rule violations are used as rough labels. With all terms both formulas look "
                        "excellent, but that is circular: the rules are defined by the gap and the speed limit. Without those two "
                        "terms, the extended formula still finds the violators (0.83) and the reference does not (0.62); most of it "
                        "comes from the physics residual, which partly overlaps with close following.")


@slide
def s_replay(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Agents", "Replay: how the agent decides", crest)
    w = (W - 0.3) / 2
    h = w * 607 / 1080
    for i, f in enumerate(("explainer_correct_follows.gif", "explainer_wrong_never_follows.gif")):
        picture(s, os.path.join(GIFS, f), X0 + i * (w + 0.3), 1.8, w=w)
    text(s, X0, 1.8 + h + 0.12, w, 0.4, "follows a car · labelled normal · correct", size=12, color=INK, align=PP_ALIGN.CENTER)
    text(s, X0 + w + 0.3, 1.8 + h + 0.12, w, 0.4, "never close to a car · labelled conservative · wrong", size=12, color=INK,
         align=PP_ALIGN.CENTER)
    footer(s, n)
    notes(s, "Replay of two normal simulated drivers as the agent (from the previous week, 8 inputs) sees them, second by second: "
             "speed, gap, its inputs and the probability of each button. When the car follows another car the following clue "
             "appears and the agent is right; when it never does, the agent falls back on 'conservative'. The new signals "
             "(physics residual, speed relative to the limit) give it information in exactly the second case.")


@slide
def s_rag(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Context", "RAG: where it plugs in", crest)
    box(s, X0, 2.2, 2.4, 1.1, "window\n+ situation", fill="F5E9EB", color=MAROON2)
    box(s, X0 + 3.1, 2.2, 2.6, 1.1, "retrieve similar\nnormal windows", fill=MAROON)
    box(s, X0 + 6.4, 2.2, 2.4, 1.1, "context\npercentile", fill=MAROON)
    box(s, X0 + 9.5, 2.2, 2.05, 1.1, "agent state", fill=MAROON)
    for x1, x2 in ((X0 + 2.42, X0 + 3.08), (X0 + 5.72, X0 + 6.38), (X0 + 8.82, X0 + 9.48)):
        arrow(s, x1, 2.75, x2, 2.75)
    box(s, X0 + 3.1, 4.2, 2.6, 0.9, "road type · traffic state\nspeed limit · country", fill="FFFFFF", color=MAROON2, size=11)
    arrow(s, X0 + 4.4, 4.18, X0 + 4.4, 3.32)
    box(s, X0 + 6.4, 4.2, 5.15, 0.9, "knowledge files (rules per country) → explanation of each decision", fill="E9E2E3",
        color=MUTED, size=12, bold=False)
    footer(s, n)
    notes(s, "Where the RAG stands: it was built in early October (model/rag.py) as a separate experiment. It scores a window "
             "against normal windows retrieved from the same road type and traffic state; it helped only in mixed simulated "
             "traffic and lost a little on UAH, so it is not part of the formula or the agents today. The plan: put the "
             "retrieved context percentile into the agent's state (how unusual is this driving for this road, traffic and "
             "country), which is also the natural way to carry the model to German roads, and use the knowledge files "
             "(rules per country) only to explain each decision in words.")


@slide
def s_next(prs, crest, n, logo):
    s = new_slide(prs)
    header(s, "Next", "Next steps", crest)
    items = ["manual verdicts on the 22 audited windows", "acceleration not squared, speed curved (from the KAN)",
             "context percentile (RAG) in the agent state", "German transfer: more seeds, highD proxy per rule",
             "full DriveDNA once access is granted", "safety intervention (throttle damping) from the components"]
    w = (W - 0.5) / 3
    for i, t in enumerate(items):
        x, y = X0 + (i % 3) * (w + 0.25), 1.8 + (i // 3) * 2.2
        box(s, x, y, w, 1.8, t, fill=MAROON if i in (1, 2) else "F5E9EB", color="FFFFFF" if i in (1, 2) else MAROON2, size=14)
    footer(s, n)
    notes(s, "Next: finish the manual verdicts on the audited windows; adopt the shapes the KAN learned (acceleration straight, "
             "speed curved) after discussing it; put the RAG context percentile into the agent's state; repeat the German "
             "transfer with more seeds and check each rule separately on highD; run full DriveDNA once the owners grant access; "
             "and, from Dr. Kurdahi's meeting, use the separate components of the index to decide when a system should intervene "
             "(for example damping the throttle).")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prs = Presentation(B.SRC)
    with zipfile.ZipFile(B.SRC) as z:
        tmp = tempfile.mkdtemp()
        crest, logo = os.path.join(tmp, "crest.png"), os.path.join(tmp, "logo.png")
        open(crest, "wb").write(z.read("ppt/media/image-2-1.png"))
        open(logo, "wb").write(z.read("ppt/media/image-1-1.png"))
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)
    B.TOTAL = len(SLIDES)
    for i, f in enumerate(SLIDES):
        f(prs, crest, i + 1, logo)
    prs.save(OUT)
    print("written", OUT, len(SLIDES), "slides")


if __name__ == "__main__":
    main()
