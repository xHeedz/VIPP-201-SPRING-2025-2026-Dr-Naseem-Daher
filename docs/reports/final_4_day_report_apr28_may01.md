# **Final 4-Day Report — Apr 28 to May 1, 2026**

**Period: **Tuesday Apr 28 – Friday May 1, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Final stretch before the Final Report submission deadline of Monday May 11. The four working days (Apr 28 – May 1) cover edits per Daher's feedback, Sections 9–11 of the report (acknowledgments, references, appendix), figure polish, the presentation deck for the May 13 talk, and the submission package.

## **Work completed**

Tightened the abstract by 32 words to 152 total, mostly by cutting redundant qualifiers and tightening the verbs. Reordered Section 5 so Social Latency mitigation precedes dose-response, per Daher's note that the mitigation is the more compelling result. Re-numbered all subsequent figures.

Added Section 9 (Acknowledgments). Re-cited Treiber 2000 in Section 2 and Bellemare 2016 in Section 3 to plug the orphan-citation gap Daher flagged.

Section 10 (References): audited the BibTeX, fixed five typos in author lists, one wrong year, two inconsistent journal abbreviations. Pruned 4 references that I had added during week-1 literature triage but never actually cited in the body. Final reference count: 33.

Section 11 (Appendix, ~4 pages). A.1: full DynamicWeightAgent and gate hyperparameters. A.2: SUMO vType configuration files reproduced verbatim. A.3: dose-response sweep results tabulated with per-seed data so the reader can recompute the standard errors.

Figure polish: regenerated the four most important figures at 300 DPI (shockwave histogram, dose-response curve, λ-sweep trade-off, architecture diagram). Matplotlib figures exported as vector PDF rather than PNG to keep them sharp at any zoom level.

Read the report aloud as a final pass — a trick that catches phrasings silent reading misses. About 30 micro-edits, plus reorganized the order of two paragraphs in Section 5.

Re-ran the end-to-end smoke test of the SUMO + ROS2 + agent stack one final time on May 1. 95/5 scenario, 5 minutes of sim time, 100 vehicles peak, both topics publishing at their target rates. Bag recorded, replayed, identical output. Tagged the final commit `term2-final-v1` so the version of the code that produced the report figures is permanently identifiable.

Compiled the final PDF: 38 pages of content + 2-page references + title page = 41 pages of report. Re-checked figures at 100% zoom (one figure had axes labels at 8 pt, bumped to 10 pt). Generated and proofread the table of contents. Spell-checked. Exported a .docx copy alongside the PDF for the Teams submission.

Built the presentation deck for the May 13 final presentation. 15-minute slot, 14 slides. Layout: title, motivation, closed-form AI, gate-head architecture, conservative-bias diagnostic + curiosity fix, SUMO setup, shockwave result, dose-response curve, Social Latency penalty + λ-sweep, fleet 24% mitigation, friction-aware AI_Weather, ROS2 stack diagram, limitations + future work, conclusion.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
