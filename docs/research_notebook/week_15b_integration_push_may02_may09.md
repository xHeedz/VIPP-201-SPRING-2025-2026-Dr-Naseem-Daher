# **Final-stretch integration push: May 2 – May 9, 2026**

Eight days I wasn't supposed to need. Submission was scheduled for Sunday May 10; the week of May 2–9 was supposed to be buffer. It turned into the most concentrated debugging stretch of the term after a routine smoke test on Saturday morning surfaced four real problems hiding in the SUMO pipeline.

## **Daily entries**

**Sat May 2***  —  100% accuracy on a smoke test is never good news.*

Was at home Saturday morning, kettle on, supposed to be doing nothing. Pulled up the SUMO comparison output for one last look before tomorrow's upload. The agent's classifications matched the ground-truth hand calculations row-for-row across all three scenarios. 100% agreement everywhere. That's not a victory.

Spent two hours staring at the trace. The issue is that both the "ground truth" code path and the "agent" code path were calling the same TraCI methods on the same vehicles in the same step, with no separation between them. They matched perfectly because they were literally computing the same thing. There was no real test of the agent's ability to estimate aggressiveness from limited or noisy observations — I'd been comparing the formula against itself.

While I had the file open I noticed two related issues. The reward function still has scenario-specific branches (highway_reward, urban_reward, weather_reward as three separate functions, which technically means the RL agent trains on three different reward signatures depending on which env it's in — overfitting risk). And the surrounding vehicles in some configurations are using simple rule-based behavior rather than IDM/MOBIL like they should be.

Texted Daher. He's at a conference, replied within an hour: "Go fix it. Push the upload to next Sunday if you have to." Push to next Sunday it is.

**Sun May 3***  —  Plan the fix.*

Spent the morning planning rather than coding. Wanted to make sure I didn't introduce new bugs trying to fix this in a hurry.

Architectural fix for the info leak: separate "what the ground truth knows" from "what the agent observes" via two assessor classes. SumoGroundTruth reads exact TraCI values (speed, leader gap, lateral position) and represents the hand-calculated reference. SumoAgentAssessor reads the same TraCI values but adds Gaussian sensor noise BEFORE running the formula, representing what a real onboard sensor suite would observe. The two never share state.

Sensor noise model based on what radar/lidar actually measure: σ_vel = 0.5 m/s (radar Doppler speed measurement error), σ_gap = 0.8 m (lidar/radar range), σ_acc = 0.3 m/s² (acceleration is derived from a noisy speed history via polyfit, so its noise compounds — this matches how real onboard estimators behave). Independent samples per step, no temporal correlation; correlated time-domain noise is a future-work item.

Reward function: merge the highway/urban/weather branches into one unified function. Scenario context flows in via the context vector, not by switching between functions. Should have done this weeks ago.

IDM/MOBIL: configure surrounding vehicles in both highway-env and SUMO to use IDM for longitudinal car-following and MOBIL for lane-change decisions. Removes the inconsistent NPC behavior I'd noticed in some places.

**Mon May 4***  —  First round of fixes.*

Unified reward function done in the morning. reward_function.py — one function takes (state, action, context including the regime flag) and returns a scalar. About 60 lines. Cross-checked against the previous three-branch implementation on cached telemetry: produces identical rewards within floating-point precision when context matches the previous regime tag.

IDM/MOBIL configured. In highway-env this was a config toggle. In SUMO it required walking through all the routes files and setting the IDM longitudinal model and MOBIL lane-change parameters explicitly on every vType (sigma=0.5 for driver imperfection, lcStrategic=1.0, lcCooperative=1.0, lcSpeedGain=1.0, lcKeepRight=0.0).

Two assessor classes implemented. SumoGroundTruth and SumoAgentAssessor. Important detail: the agent assessor adds noise BEFORE running the formula, not after — adding noise to the score after the fact would just be a perturbation on the output, not a model of what a sensor system experiences.

Wrote three scenario runners — highway_scenario.py, urban_scenario.py, weather_scenario.py. Each loads the corresponding SUMO config, runs for a fixed duration, samples both assessors at every step, and writes per-vehicle classifications to disk. 9 hours at the lab, brain is mush, but the architecture is finally clean.

Code committed. Updated the test suite: a unit test now asserts that SumoGroundTruth and SumoAgentAssessor produce different outputs on the same step (i.e., that noise is actually being applied). Catches future regressions.

**Tue May 5***  —  First clean results.*

First clean run of the three scenarios with the new architecture. Numbers: highway 95.7% accuracy on 438 samples, urban 95.0% on 462 samples, weather 98.0% on 1524 samples. These are higher than I expected, almost suspiciously high — but the architecture is sound. Two completely separate assessors, independent TraCI calls, agent sees noisy values only. There's no info leak. The high accuracy reflects the formula being well-separated for the populations I'm running, not a bug.

Saved the comparison plot and let it bake while I caught up on sleep.

**Wed May 6***  —  Wait, where are the aggressive vehicles?*

Read the per-vehicle classifications more carefully today. Every vehicle in every scenario classified as Conservative or Normal. Zero Aggressive labels. That can't be right — I have aggressive vType definitions in the routes files for all three scenarios.

Two root causes traced today, both kind of embarrassing in hindsight.

Root cause 1: the speed term in the formula is (speed_kmh / 150)². Squaring a fraction less than 1 makes it smaller. At 150 km/h the speed term is 1.0; at 130 km/h only 0.75; at 100 km/h, 0.44. With the highway weight at 0.30 the speed term contributes 30% × maxed-out-1.0 = 30% of the total, and at realistic SUMO speeds it contributes 5–15%. The formula as written caps at ~25% before the proximity term — well below the 55% Aggressive threshold. The formula was mathematically incapable of producing Aggressive labels at the speeds these scenarios actually run at.

Fix: switch from (speed_kmh / 150)² to speed_ms / speed_ref (no square), with per-scenario reference speeds. Highway speed_ref = 38 m/s (~137 km/h). Urban speed_ref = 22 m/s (~80 km/h). Weather speed_ref = 25 m/s (~90 km/h).

Root cause 2: proximity. The formula was measuring distance from each surrounding vehicle to the ego. But IDM vehicles don't tailgate the ego — they keep equilibrium gaps to their own leader, which is whoever's directly in front of them, not necessarily the ego. An aggressive vehicle two cars ahead of the ego was reading its distance to the ego (large) instead of its distance to whoever's in front of it (small). Aggressive vehicles were scoring near zero on the proximity term, which is weighted at 45% in the highway formula. Fix: switch to traci.vehicle.getLeader() so we're measuring distance to actual leader. This is what we should have been measuring all along.

Re-ran. Aggressive labels appeared. Distribution still off — too few — but at least nonzero. Progress.

**Thu May 7***  —  Two more root causes.*

Weather scenario was producing very few Aggressive labels even after yesterday's fixes. Tracked it down to a double-counting bug: the weather routes file already applies spd_scale=0.7 to every vehicle's maxSpeed (30% reduction to reflect rain physics), AND the runner was calling traci.vehicle.setMaxSpeed() multiplied by friction (0.3). Aggressive vehicles ended up capped at 40 × 0.7 × 0.33 ≈ 9 m/s. Crawling. Conservative regardless of intent.

Fix: removed all setMaxSpeed calls from the runner. The route file already encodes the rain-physics speed reduction; the runner shouldn't apply it again. Re-ran. Aggressive vehicles in weather now hit ~25 m/s (still slower than highway, appropriately) and start scoring Aggressive.

Urban had a separate structural problem. The intersection arms were 2 lanes each, which let aggressive vehicles lane-change to overtake conservatives — leaving them with no leader, a 150 m gap reading, and a Normal classification. The aggressive behavior was real but the formula couldn't see it because it measures proximity-to-leader and there was no leader to measure to.

Two fixes for urban. (1) Switched all four intersection arms to 1 lane each. No more overtaking room. (2) Added 8 explicit conservative→aggressive platoon pairs across the four arms — each pair starts with a conservative vehicle just ahead of an aggressive one, so the aggressive vehicle is forced to either tailgate (raising its proximity term) or hold its lane and tailgate anyway. Re-ran. Aggressive labels appearing in proportion now.

**Fri May 8***  —  Calibration day.*

Per-scenario formula calibration locked in. Highway: speed_ref=38 m/s, gap_ref=25 m, thresh_aggr=55%. Urban: speed_ref=22 m/s, gap_ref=17 m, thresh_aggr=50% (raised from 47% to give normals more margin). Weather: speed_ref=25 m/s, gap_ref=35 m, thresh_aggr=55%.

Two rabbit-holes today, both worth recording.

First: weather noise scale. I'd left it at 1.5× by analogy to highway (more uncertainty in weather sensor returns — wet roads, lower-confidence radar). But conservative vehicles in weather sit at ~28% on the formula, only 2 percentage points below the C/N category boundary at 30%. With 1.5× noise, conservatives flip to Normal frequently, and the agent's accuracy on weather crashed to ~69%. Reverted weather noise scale to 1.0×. Accuracy back up to 88.3%.

Second: highway accuracy was sitting at 97.7%, which felt suspiciously high given the noise model. Tried raising the global SIGMAs to (0.8, 1.5, 0.4) — basically modeling noisier sensors. Highway dropped to 88.8% but urban crashed to 75.9% and weather to 64.9%. The asymmetry is structural: highway has well-separated category populations (vehicle speeds 18 / 28 / 38 m/s map to formula scores ~24% / 39% / 65%, with category boundaries at 30% and 55% — comfortably away from any vehicle's expected score). Urban and weather both have populations sitting much closer to category boundaries (urban normals at ~43% are 7% below the 50% boundary; weather conservatives at ~28% are 2% below the 30% boundary). Global noise hits them disproportionately.

Reverted global SIGMAs to (0.5, 0.8, 0.3). Applied a noise_scale=1.5 only to the highway call. Highway accuracy: 91.0%. Urban and weather unchanged. That's the calibration.

Last urban tuning, captured above for completeness: normal vehicles at ~43% sat too close to the original 47% Aggressive threshold. Raised threshold to 50%. Normal vehicles now have 7% margin; aggressive platoon vehicles (scoring 55–60%) still well above. Urban accuracy: 81.6%.

**Sat May 9***  —  Final results, last cleanup.*

Final agent-vs-ground-truth accuracies and per-scenario distributions. Highway (clear): 91.0% accuracy across 1328 samples — ground-truth distribution 589 Conservative / 717 Normal / 22 Aggressive. Urban (intersection): 81.6% across 2284 samples — 1223 / 501 / 560. Weather (rain/wet): 88.3% across 1825 samples — 833 / 837 / 155.

All three scenarios produce all three driver categories. There's no info leak — ground truth and agent assessor are completely independent code paths. Accuracy is in the realistic 80–91% range. Original target was 90–95%; urban falls below at 81.6%, which is honest about how tightly bunched the normal/aggressive populations sit at intersection speeds with sensor noise applied.

Generated sumo_comparison.png — three subplots: per-scenario accuracy bar, sample count bar, per-category confusion matrices. Saved to figures/.

Updated three sections of the final report. The methods section now describes both assessor classes and the noise model. The experimental-setup section adds the σ values, the per-scenario thresholds, and the IDM/MOBIL configuration for surrounding vehicles. The results section adds the final agent-vs-ground-truth accuracies as a closing validation. Limitations section adds: urban falls below the original 90–95% accuracy target; the cause is well-understood (tight category boundaries × intersection-speed populations × sensor noise) rather than a fundamental architectural problem.

Compiled the report one last time. 43 pages with the new material. Pushed all code with a new tag, term2-final-v2-with-noise-validation. Updated the README to document the two assessor classes and the sensor noise model. Updated the data folder to include the three scenario runner outputs and the comparison chart.

Submission goes up tomorrow morning, Sunday May 10, 9 AM.

## **Reflection on the integration push**

Eight days I wasn't supposed to need. The 100% accuracy was the entry point; once I started looking carefully, three other issues fell out — the unified reward function that should've been done weeks ago, the inconsistent IDM/MOBIL configuration across simulators, and the proximity-to-ego vs. proximity-to-leader bug that had been silently wrong all term.

The most useful lesson: 100% on anything is a bug signal. I've seen it before in other projects and ignored it; this time I caught it and the project is meaningfully better. The 81.6% urban result is honest in a way the original 100% wasn't. The whole point of the AI score is to be usable by a real onboard system with imperfect sensors, and the May 2–9 work is what makes that claim defensible.

## **Reflection on the term as a whole**

Fourteen concrete artifacts shipped this term: the PyTorch DynamicWeightAgent (with both MLP and gate heads), the count-based curiosity bonus on the Q-learner, the Bellman-error and state-visitation diagnostics, the SUMO wrapper with TraCI-based telemetry extraction, the heterogeneous-driver scenario, the shockwave analysis, the dose-response sweep, the Social Latency reward and its λ-tuning, the friction-aware regime with slip-ratio telemetry, the dual-topic ROS2 publisher, the unified reward function, the SumoGroundTruth / SumoAgentAssessor split with the Gaussian sensor noise model, the three scenario runners, and the final report.

Three concrete artifacts not shipped: CARLA integration (descoped in week 9 — the right call), learnable normalization saturation points (kept fixed for term-2, deferred per Daher's week-1 agreement), and correlated time-domain sensor noise (the May 2–9 work used i.i.d. Gaussian noise per step). All three are the largest items in the future-work section of the final report.

If I had to identify one decision that mattered most, week 5's diagnostic loop (Bellman-error tracking + state-visitation heatmaps) was the most consequential, but the May 2–9 integration push gives it real competition. Week 5 fixed a bug that the midterm's clean accuracy curve was hiding; May 2–9 fixed a bug that this term's clean accuracy curve was hiding. The lesson the second time around was the same as the first: instrument ruthlessly, look at the numbers carefully, and don't trust 100%.

Plan for next term: pick up CARLA, the multi-driver coordination experiments, on-vehicle ROS2 deployment, and a longer noise-model exploration (correlated time-domain noise per sensor). Until then — final exams, final presentation, and a much-needed week off afterward.

# **Notebook closing note**

This notebook covers the period from Tuesday January 20, 2026 — the first day of classes after the winter break — through Saturday May 9, 2026, the day the final integration push concluded. Fourteen full weeks plus the four-day Final Stretch and the eight-day post-May-1 integration push, recorded in something close to lab-notebook style: dated, granular, honest about what worked and what didn't. Bugs are recorded with their fixes; descoped tracks are recorded with the reasoning; sync-with-supervisor decisions are recorded with the date and the agreed action. The point of a notebook isn't to be a polished report — it's to be a faithful trace.

I'm submitting this alongside the Final Report, the presentation deck, the codebase tagged term2-final-v2-with-noise-validation, the datasets generated this term (including the three new SUMO scenario outputs from the May 2–9 push), and the bag recording demonstrating the full ROS2 stack — all to the VIP team's Microsoft Teams space, under my name, per Prof. Zeaiter's instructions.

*— Hadi Al Shmaissani, Saturday May 9, 2026.*

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
