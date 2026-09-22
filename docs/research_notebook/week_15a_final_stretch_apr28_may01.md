# **Final stretch: Tuesday Apr 28 – Friday May 1, 2026**

The last four working days before the Final Report submission deadline of Monday May 11. (The deadline is technically May 11 but I want my submission ready by May 1 so the weekend of May 2-3 is buffer for any last-minute fixes, and the week of May 4-10 is for Final Presentation prep on May 13.) These four days are about edits, the back-matter sections, the deck, and the submission package.

## **Daily entries**

**Tue Apr 28***  —  Edits per Daher's feedback + the abstract.*

Started the day going through Daher's TODO list from the Monday sync. The abstract tightening was the most fiddly — I needed to cut about 30 words while preserving the four-contribution structure. Ended up at 152 words from the original 184, mostly by removing redundant qualifiers ("specifically," "in particular") and tightening the verbs. Read it three times in a row, decided it's tight enough.

Reordered Section 5 per the feedback. Social Latency mitigation now comes before dose-response, which means the reader sees the mitigation result first (compelling, practical), then the dose-response (the empirical justification for the mitigation). Story flows better this way. Had to re-renumber all the figures — annoying but mechanical.

Added the acknowledgments section (Section 9). Thanked Prof. Daher for supervision, the VIP team for the lab access and discussion, and the Anthropic / Meta / OpenAI engineers whose published architectural choices indirectly informed the gate-head design. Kept it brief — one paragraph.

Re-cited Treiber 2000 in Section 2 (the original IDM paper, which I'd been leaning on for the shockwave interpretation but hadn't actually cited in the related-work section), and Bellemare et al. 2016 in Section 3 (the count-based exploration paper, which was cited in the methods but I'd missed it in the related-work bibliography). Now there are no orphan citations and no lean-on-without-citing instances. Clean.

**Wed Apr 29***  —  References, appendix, figures.*

Section 10: references. Audited the BibTeX file. ~37 entries. Five had typos in author lists, one had the wrong year, two had inconsistent journal abbreviations. Fixed all. Also pruned 4 entries that I'd added during literature triage in week 1 but never actually cited in the body text — keeping them in the bibliography would be padding. Final count: 33 references.

Section 11: appendix. Three subsections. (A.1) The exact gate-network and DynamicWeightAgent architectures with all hyperparameters. (A.2) The vType configuration files for the SUMO scenarios, reproduced verbatim. (A.3) The full dose-response sweep results tabulated, with seed-level data so the reader can recompute the standard errors. About 4 pages of appendix.

Figure polish: regenerated the four most important figures at 300 DPI for print quality. The shockwave histogram, the dose-response curve, the λ-sweep trade-off, and the architecture diagram. The TikZ ones were already vector and didn't need re-rendering. The matplotlib ones I exported as PDF rather than PNG, which keeps them sharp at any zoom level.

**Thu Apr 30***  —  Final-pass read-aloud + presentation deck.*

Did a final read-through of the whole report. Read it aloud — a trick I picked up last term, catches phrasings that silent reading misses. Made about 25 micro-edits (commas, awkward phrasings, two paragraphs that needed to be split). The end-to-end argument now flows cleanly: motivation → architecture → SUMO experiments → reward shaping → friction → ROS2 → limitations → future work → conclusion.

Started the presentation deck for the May 13 talk. 15-minute slot, so target ~12 slides plus title and Q&A holder. Layout: (1) title; (2) the practical motivation (autonomous + ADAS need to detect aggressive neighbors); (3) closed-form AI formulation; (4) gate-head architecture; (5) the conservative-bias diagnostic and curiosity fix (one slide for the punchline pair of figures); (6) SUMO setup; (7) shockwave result; (8) dose-response curve; (9) Social Latency penalty + λ-sweep; (10) fleet result with the 24% mitigation; (11) friction-aware AI_Weather; (12) ROS2 stack diagram; (13) limitations + future work; (14) conclusion + questions.

About 6 hours on the deck, mostly on getting the figures to be legible at projection size. The dose-response curve and the shockwave histogram are the slides I want the audience to remember, so I gave them extra real estate and reduced the text on those slides to a single sentence each.

**Fri May 1***  —  Final revision + smoke test + submission package.*

Started the day reading the report aloud one more time, this time at the lab so I could check formatting issues on a different monitor. Made about 30 wording edits and reorganized the order of two paragraphs in Section 5. The argument now flows the way I want: motivation → architecture → experiments → reward shaping → friction → ROS2 → limitations → future work → conclusion. Solid.

Re-ran the end-to-end smoke test of the SUMO + ROS2 + agent stack one final time, just to make sure nothing has bit-rotted in the last week of writing-only work. 95/5 scenario, 5 minutes of sim time, 100 vehicles peak, both topics publishing at their target rates. Bag recorded, replayed, identical output. Tagged the final commit `term2-final-v1` so the version of the code that produced the report figures is permanently identifiable.

Compiled the final PDF. 38 pages of content + 2-page references + title page = 41 pages of report. Re-checked figures for legibility at 100% zoom (one figure had axes labels at 8pt — bumped to 10pt). Generated and proofread the table of contents. Spell-checked. Exported a copy as a .docx for the Teams submission alongside the PDF.

End-of-day, end-of-term: the deliverables checklist as it stands today. Notebooks (this document) — uploading to the Teams space alongside scanned versions of my paper notebook. Peer reviews — completed and emailed to Prof. Zeaiter directly per his instructions. Final report — done, scheduled for upload Sunday May 10. Final presentation — slide deck done, 15-minute slot booked for Wednesday May 13. Codebase — pushed to the lab GitHub with the v1 tag, README updated, requirements pinned. Datasets — driving_behaviors_dataset.csv (legacy), demo_data.csv (midterm), the SUMO scenario configs (term-2), and a 5-minute bag recording showing the full ROS2 stack — all uploaded to the Teams space.

Closing thoughts on the term. The work split roughly in half between agent-side machine learning (weeks 1-7) and systems / scale work (weeks 8-14). In retrospect, that division was the right one. The gate-head agent and the curiosity-augmented Q-learner aren't particularly novel as machine-learning artifacts — they're textbook applications of well-understood techniques to a specific problem. What is novel is the pipeline: a closed-form, interpretable Aggressiveness Index that feeds an RL fine-tuning loop that produces a regime-aware predictor that publishes on a ROS2 topic that downstream planners can subscribe to. The interpretability survives all the way through. Most autonomous-driving stacks lose interpretability somewhere between perception and planner; ours doesn't, and that's the contribution worth defending in the final presentation.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
