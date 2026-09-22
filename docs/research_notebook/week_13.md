# **Week 13: Tuesday Apr 14 – Monday Apr 20, 2026**

**Theme — ***Summary topic + final report drafting kicks off (Sections 1-3).*

## **Goals for the week**

Two parallel tracks. (1) Add the /env/aggressiveness_summary topic publishing at 1Hz with population-level aggregates: mean_ai, max_ai, neighbor_affected_count, fraction_above_aggressive_threshold. Use a small aggregator node that subscribes to /env/aggressiveness_index and publishes the summary on a timer. (2) Draft Sections 1-3 of the final report: introduction, related work, methods. Target page count for the three sections combined: ~12 pages.

## **Daily entries**

**Tue Apr 14***  —  Summary topic.*

Wrote SummaryNode in summary_node.py. On startup, creates a 1Hz timer. Maintains an in-memory dict of vehicle_id → most-recent (timestamp, ai_score, mixture_weights). On each timer tick, evicts entries older than 2 seconds (so dropped vehicles don't pollute the summary), computes population statistics, publishes on /env/aggressiveness_summary. Custom message: AggressivenessSummary.msg with fields float32 mean_ai, float32 max_ai, int32 n_vehicles, int32 n_affected_neighbors (count of vehicles whose mean over the last 2s exceeds reference + 1σ), float32 aggressive_fraction.

Tested with 50-vehicle fleet running the heterogeneous SUMO scenario. Summary topic publishes at exactly 1Hz, message size is small (~32 bytes), and the reported aggregates track the ground-truth aggregates from the SUMO trajectory log (cross-checked offline). n_affected_neighbors fluctuates between 8 and 14 in the 5%-aggressive scenario, consistent with the spatial-decomposition results from week 8.

**Wed Apr 15***  —  Final report: Section 1 (introduction).*

Started the final report. Working from the LaTeX template Daher's group standardized on. Section 1 (Introduction): about 2 pages. Three subsections: (1.1) the practical problem — autonomous and ADAS systems need a real-time, interpretable measure of how aggressive surrounding drivers are, both for the ego car's own situational awareness and for fleet-level coordination; (1.2) what's missing in the literature — most behavior-classification work is post-hoc and uses opaque deep-learning classifiers, which makes the output hard to audit and combine with downstream planners; (1.3) what this project contributes — a closed-form interpretable AI score, a context-aware weight predictor, an empirical study of population-level shockwaves, and a reward-shaping intervention that mitigates them.

Wrote the section in one sitting. ~1800 words. Will iterate next week.

**Thu Apr 16***  —  Final report: Section 2 (related work).*

Section 2: about 3 pages. Three threads: (2.1) classical behavior-classification approaches — SVM-based, K-means clustering on telemetry features, all the work pre-2017ish; (2.2) deep-learning behavior-classification — LSTM-based and transformer-based classifiers, mostly post-2018, the work that's high-accuracy but opaque; (2.3) interpretable / hybrid approaches — including the work on explicit aggressiveness indices in transportation engineering (Murphey 2009, Brito 2020) which is the closest neighbor to what we're doing.

Spent a chunk of the day re-reading half of the ~30 papers I have in Zotero to make sure I'm citing them correctly and not misrepresenting their claims. Citing papers I haven't really read is exactly the kind of small academic dishonesty I want to avoid, even by accident.

**Fri Apr 17***  —  Final report: Section 3 (methods, part 1).*

Section 3 is the long one — methods. Started with 3.1 (the closed-form AI formulation) and 3.2 (normalization functions for the four telemetry channels). Mostly lifted from the midterm with cleaner equations and tighter prose. About 3 pages so far.

The equations look clean in LaTeX. AI = 100 · Σᵢ wᵢ · Nᵢ. Each Nᵢ has its own piecewise saturation form, equation-block presentation. Took the time to render each one as a numbered display equation rather than inline.

AI(s) = 100 · [w_speed · N_speed(s) + w_accel · N_accel(s) + w_prox · N_prox(s) + w_wave · N_wave(s)]

**Sat Apr 18***  —  Off (mostly).*

Off mostly. Did about an hour of editing on Section 1 — tightened the introduction's claim of contribution because the original draft over-promised on the CARLA / urban-occlusion results we never actually did.

**Sun Apr 19***  —  Final report: Section 3 (methods, parts 2-3).*

Section 3.3 (DynamicWeightAgent architecture, both heads) — 2 pages, includes the gate-head equations and the diagram of the architecture I made today. 3.4 (count-based curiosity bonus) — 1 page, with the discretization and the β-tuning derivation. 3.5 (Social Latency penalty) — 1 page, with the reward formula and the rolling-baseline definition.

Re-drew the architecture diagram in TikZ — the previous Inkscape version had pixelation issues at the print resolution Daher's group uses for submitted reports. Took longer than I wanted but the result is crisp.

**Mon Apr 20***  —  Sync.*

Sync. Walked Daher through the Section 1-3 drafts. He read the abstract live (one paragraph I'd added at the top of the document) and immediately said it was over-claiming on the SUMO multi-driver experiments — "You don't have CARLA, you don't have real-world data. Be modest." He's right. Edited the abstract in the meeting from a 200-word draft down to about 150 words with more conditional language: "in simulation," "in our heterogeneous-driver experiments," etc.

Two requests for week 14: (a) finish Section 4 (experimental setup), Section 5 (results — the heart of the paper), Section 6 (discussion), and Section 7 (limitations + future work) by next Monday. (b) Run an end-to-end smoke test on the full ROS2 stack one more time before we finalize, just to make sure nothing's bit-rotted.

## **Reflection**

Drafting the report is going faster than I expected, partly because the term's been documented well in this notebook and the figures are already produced. The summary topic took half a day, the rest of the week was writing. The hardest piece was Section 2 — making sure I'm representing the literature accurately. Daher's note about not over-claiming in the abstract is the kind of feedback I'm going to need to listen to more carefully as the report comes together.

## **Plan for next week**

Sections 4-7. Section 4 is experimental setup (SUMO scenarios, hyperparameters, hardware, evaluation protocol). Section 5 is the results — shockwave + dose-response + Social Latency mitigation + friction-aware AI_Weather. Section 6 is discussion, Section 7 is limitations and future work (with CARLA descoping mentioned).

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
