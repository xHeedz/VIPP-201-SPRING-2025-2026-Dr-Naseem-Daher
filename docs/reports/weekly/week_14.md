# **Weekly Report — Week 14**

**Period: **Tuesday Apr 21 – Monday Apr 27, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Draft Sections 4–8 of the final report. Run an end-to-end smoke test of the full ROS2 + SUMO + agent stack to confirm nothing has broken during the report-writing phase.

## **Work completed**

Section 4 (Experimental Setup, ~3 pages). Subsections 4.1 (highway-env baseline), 4.2 (SUMO heterogeneous-driver scenario with full vType definitions, demand schedule, friction config), 4.3 (training hyperparameters), 4.4 (evaluation protocol). Two configuration tables with bolded differences for reproducibility.

Section 5 (Results, ~6.5 pages). 5.1 population-level shockwave (per-vehicle AI distribution, spatial-bin shockwave histogram, t-test). 5.2 dose-response curve with the saturating-exponential fit overlaid. 5.3 Social Latency mitigation (λ-sweep, behavioral inspection, fleet result, travel-time cost). 5.4 friction-aware AI_Weather. Per Daher's feedback, reordered so Social Latency comes before dose-response — the mitigation is the more compelling result.

Sections 6 (Discussion, ~2 pages), 7 (Limitations + Future Work, ~2 pages), 8 (Conclusion, ~1 page).

End-to-end smoke test on the full stack: SUMO → ROS2 telemetry publisher → AgentNode → /env/aggressiveness_index → SummaryNode → /env/aggressiveness_summary. Heterogeneous 5%-aggressive scenario, 5 minutes. Bag recorded, replayed, output identical. Per-vehicle AI scores match the SUMO trajectory log (max diff 0.02 on per-vehicle means, attributable to floating-point summation order).

Read the entire report top-to-bottom (about 38 pages with figures) on Sunday. Found ~30 small issues: typos, awkward phrasings, two figure-caption mismatches, one citation pointing to the wrong paper. All fixed.

## **Key results**

All five sections (4–8) drafted to roughly 14.5 combined pages. Total report length now ~38 pages.

Smoke test green: full stack reproducible, no bit-rot from the writing-only weeks.

## **Plan for next week**

Final 4 days (Apr 28 – May 1). Edits per Daher's feedback from the Monday sync (tighten the abstract by another 30 words, move Social Latency mitigation ahead of dose-response, add an acknowledgments section, re-cite Treiber 2000 and Bellemare 2016 in Section 2). Sections 9–11 (acknowledgments, references, appendix). Final figure polish at print resolution. Presentation deck for the May 13 talk. Submission package for Teams.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
