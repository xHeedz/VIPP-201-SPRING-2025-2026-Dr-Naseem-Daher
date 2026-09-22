# **Week 14: Tuesday Apr 21 – Monday Apr 27, 2026**

**Theme — ***Sections 4-8 of the final report. End-to-end smoke test. The home stretch.*

## **Goals for the week**

Draft Sections 4-8 of the final report. Section 4 is experimental setup. Section 5 is the results section — shockwave, dose-response, Social Latency, friction-aware. Section 6 is discussion. Section 7 is limitations and future work. Section 8 is the conclusion. Plus an end-to-end smoke test on the full ROS2 + SUMO + agent stack to confirm nothing has broken during the report-writing phase.

## **Daily entries**

**Tue Apr 21***  —  Section 4 (experimental setup).*

Section 4. Subsections: 4.1 (highway-env baseline scenario, the one carried over from term-1), 4.2 (SUMO heterogeneous-driver scenario with full hyperparameters: vType definitions, demand schedule, friction config), 4.3 (training hyperparameters for the gate-head agent and Q-learner), 4.4 (evaluation protocol: held-out scenarios, seed handling, reference-baseline definitions). About 3 pages with two configuration tables.

The vType configuration table (normal vs. aggressive parameters, friction settings) gets a lot of weight in this section because reproducibility hinges on those parameters. Made it as a clean LaTeX table with bolded differences. Worth the extra hour.

**Wed Apr 22***  —  Section 5 (results) — shockwave + dose-response.*

Section 5 is going to be the longest section of the report. Started with 5.1 (population-level shockwave). Two figures: per-vehicle AI distribution colored by vehicle type (the cleanly-separated aggressive cluster), and the spatial-bin shockwave histogram (close vs. medium vs. far normals). Two tables: per-class summary statistics and the t-test outputs.

Then 5.2 (dose-response curve). One figure: aggressive-fraction sweep with the saturating-exponential fit overlaid. The fit equation in the caption: AI(f) = 26.3 + 12.5·(1 - exp(-f/0.06)), R² = 0.998. Discussion paragraph emphasizes the practical implication — small fractions of aggressive drivers do most of the population-level damage.

About 3.5 pages of section so far.

**Thu Apr 23***  —  Section 5 — Social Latency + friction-aware.*

Continued Section 5. 5.3 (Social Latency mitigation): λ-sweep figure, behavioral-inspection plots (following distance + lane-change rate), 50%-fleet population-level result, trade-off discussion (24% mean-AI reduction at +2.8% travel-time cost). 5.4 (friction-aware AI_Weather): low-friction scenario results, weather-mixture rise from ~5% to ~50% under elevated slip, AI_Weather statistics under low-friction with and without aggressors.

About 3 more pages. Section 5 is now 6.5 pages, which is the right size for the central results section of a 35-page report.

**Fri Apr 24***  —  Sections 6 (discussion) + 7 (limitations / future work).*

Section 6 is the discussion. Three threads: (a) what the closed-form AI score buys us that opaque classifiers don't — interpretability, modularity, the ability to compose with reward-shaping; (b) why the dose-response curve's saturation matters for ADAS deployment policy — even partial fleet penetration of the Social Latency policy recovers most of the population-level damage; (c) a brief positioning against the deep-learning literature with a note that we're not claiming higher accuracy than transformer-based behavior classifiers, we're claiming a different operating point on the interpretability-accuracy trade-off.

Section 7: limitations. (a) Synthetic SUMO traffic, no real-world telemetry. (b) CARLA descoped — would have given us occlusion-aware urban telemetry, which is a real gap. (c) The hand-picked target weights remain a limitation; the embedding-drift result in week 4 hints that a longer RL fine-tuning phase could shift them but we didn't pursue it. (d) The 30m proximity radius for the Social Latency penalty is a magic constant — should be a learned function of vehicle speed in future work.

**Sat Apr 25***  —  End-to-end smoke test + Section 8 conclusion.*

End-to-end smoke test on the full stack — SUMO → ROS2 telemetry publisher → AgentNode → /env/aggressiveness_index → SummaryNode → /env/aggressiveness_summary. Ran the heterogeneous 5%-aggressive scenario for 5 minutes. Recorded a bag. Replayed it. Output is reproducible. AI scores match SUMO trajectory log when computed offline (max diff 0.02 on per-vehicle means, attributable to floating-point summation order). Stack is healthy.

Wrote Section 8 (conclusion) — about 1 page summarizing the four contributions in a plain-language way, with a closing paragraph on the next-term roadmap (CARLA, multi-driver coordination experiments, on-vehicle ROS2 deployment).

**Sun Apr 26***  —  Pass through the whole report.*

Read the whole report top to bottom. About 38 pages with figures. Found maybe 30 small issues — typos, awkward phrasings, two figure captions that didn't match the figure numbers in the text, one citation pointing to the wrong paper. Fixed all of them. Made a list of bigger questions for Daher: (a) is the abstract strong enough? (b) does Section 6 over-claim on interpretability? (c) is the conclusion the right length?

**Mon Apr 27***  —  Sync — final review pass.*

Sync was 45 minutes today, longer than usual because we went through the whole report. Daher had detailed feedback, recorded as TODO list. Most important items: (a) tighten the abstract by another 30 words; (b) move the Social Latency mitigation discussion ahead of the dose-response in Section 5 because the mitigation is the more compelling result; (c) add an acknowledgments section thanking the VIP team; (d) re-cite a couple of papers I'd leaned on (Treiber 2000, Bellemare 2016) in Section 2 because the related-work coverage felt thin in two places. He approved the rest.

End of the term-of-work. Next four working days: edits, Sections 9-11 (acknowledgments, references, appendix), polish, submit.

## **Reflection**

Five sections drafted in a week, plus a successful end-to-end smoke test. The pace was sustainable because the term's results were already in hand and the figures were already produced — the writing was mostly transcription with structure. Daher's review was thorough but didn't ask for any structural rewrites, which means the architecture of the report is sound. The next four days are detail work, not new content. Feels manageable for the first time in the term's late stages.

## **Plan for next week**

Final 4 days (Apr 28 – May 1): edit per Daher's feedback, Sections 9-11 (acknowledgments, references, appendix), final figure polish, presentation deck for the May 13 talk, final submission.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
