# **Weekly Report — Week 15 (Post-deadline integration push)**

**Period: **Saturday May 2 – Saturday May 9, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Originally, the week between May 1 (work-complete) and May 10 (Teams upload) was scheduled as buffer. A routine final smoke test on Saturday May 2 surfaced a 100% agent-vs-ground-truth accuracy, which is a textbook bug signal. Investigation surfaced four real problems in the SUMO pipeline that needed to be fixed before submission: (1) an info leak between the agent's classifier and the ground-truth reference; (2) scenario-specific reward functions that should have been unified; (3) inconsistent IDM/MOBIL configuration on surrounding vehicles across simulators; (4) a structural inability of the formula to produce Aggressive labels at the speeds these scenarios actually run at. Prof. Daher approved pushing the upload to Sunday May 10.

## **Work completed**

Architectural fix for the info leak: introduced two separate assessor classes. SumoGroundTruth reads exact TraCI values (speed, leader gap, lateral position) and represents the hand-calculated reference. SumoAgentAssessor reads the same TraCI values but adds Gaussian sensor noise before running the formula, representing what a real onboard sensor suite would observe. Critically, noise is applied to the inputs rather than to the output score — a perturbation on the score after the fact would not be a model of what a sensor system actually experiences. A unit test asserts the two classes produce different outputs on the same step, catching future regressions.

Sensor noise model: σ_vel = 0.5 m/s (radar Doppler speed measurement error), σ_gap = 0.8 m (lidar/radar range), σ_acc = 0.3 m/s² (acceleration is derived from a polyfit on the noisy speed history, so its noise compounds — matches how real onboard estimators behave). Independent samples per step, no temporal correlation; correlated time-domain noise is a future-work item.

Unified reward function (reward_function.py): merged the highway / urban / weather branches into one function that takes (state, action, context including the regime flag) and returns a scalar. About 60 lines. Cross-checked against the previous three-branch implementation on cached telemetry: produces identical rewards within floating-point precision when context matches the previous regime tag.

IDM/MOBIL configuration: walked through all SUMO routes files and set the IDM longitudinal model and MOBIL lane-change parameters explicitly on every vType (sigma=0.5 driver imperfection, lcStrategic=1.0, lcCooperative=1.0, lcSpeedGain=1.0, lcKeepRight=0.0). Equivalent toggle in highway-env.

Three scenario runner scripts written: highway_scenario.py, urban_scenario.py, weather_scenario.py. Each loads the corresponding SUMO config, runs for a fixed duration, samples both assessors at every step, and writes per-vehicle classifications to disk.

## **Four root causes addressed**

Root cause 1 — speed term mathematically incapable of producing Aggressive labels. The formula used (speed_kmh / 150)². Squaring a fraction less than 1 makes it smaller. Combined with the highway weight 0.30, the speed term contributed at most 30% of the total at full-saturation — well below the 55% Aggressive threshold. Fix: switch from (speed_kmh / 150)² to speed_ms / speed_ref (no square), with per-scenario reference speeds. Highway speed_ref = 38 m/s, urban speed_ref = 22 m/s, weather speed_ref = 25 m/s.

Root cause 2 — proximity measured to the ego instead of to the actual leader. IDM vehicles don't tailgate the ego; they keep equilibrium gaps to whoever is directly in front of them. An aggressive vehicle two cars ahead of the ego was reading its (large) distance to the ego instead of its (small) distance to its actual leader. Fix: switch to traci.vehicle.getLeader() so the formula measures distance to the actual following relationship.

Root cause 3 — weather double-counting on the speed reduction. The weather routes file already applied spd_scale=0.7 to all vehicle maxSpeeds (rain physics), and the runner also called traci.vehicle.setMaxSpeed() multiplied by friction (0.3). Aggressive weather vehicles ended up capped at 40 × 0.7 × 0.33 ≈ 9 m/s — crawling — and scored Conservative regardless of intent. Fix: removed all setMaxSpeed calls from the runner.

Root cause 4 — urban structural problem. The intersection arms had 2 lanes each, which let aggressive vehicles lane-change to overtake conservatives, leaving them with no leader and a 150 m proximity reading. Fix: switched all four intersection arms to 1 lane each, and added 8 explicit conservative→aggressive platoon pairs distributed across the four arms so the aggressive vehicles are forced into tailgating relationships from the start.

## **Per-scenario calibration**

Final calibration parameters: highway speed_ref=38 m/s, gap_ref=25 m, thresh_aggr=55%, noise_scale=1.5; urban speed_ref=22 m/s, gap_ref=17 m, thresh_aggr=50%, noise_scale=1.0; weather speed_ref=25 m/s, gap_ref=35 m, thresh_aggr=55%, noise_scale=1.0. The urban threshold was raised from 47% to 50% to give Normal vehicles (scoring ~43%) more margin. Highway noise_scale was raised to 1.5 only after a global SIGMA increase to (0.8, 1.5, 0.4) crashed urban accuracy to 75.9% and weather to 64.9% — those scenarios have populations sitting much closer to category boundaries and global noise hits them disproportionately. Targeted highway-only noise_scale was the cleaner fix.

## **Key results**

Final agent-vs-ground-truth accuracies and per-scenario distributions: Highway (clear) 91.0% across 1328 samples, ground-truth distribution 589 Conservative / 717 Normal / 22 Aggressive. Urban (intersection) 81.6% across 2284 samples, 1223 / 501 / 560. Weather (rain/wet) 88.3% across 1825 samples, 833 / 837 / 155.

All three scenarios produce all three driver categories. Ground truth and agent assessor are completely independent code paths — no info leak. Accuracy is in the realistic 80–91% range. The original target was 90–95%; urban falls below at 81.6%, attributable to the tight category boundaries × intersection-speed populations × sensor noise interaction (well-understood, not a fundamental architectural issue).

Generated sumo_comparison.png — three subplots covering per-scenario accuracy, sample counts, and per-category confusion matrices.

## **Issues encountered**

The 100% accuracy signal is the entry point for the entire week; everything else followed from looking carefully at why that number was suspicious. Two specific tuning rabbit-holes worth flagging in retrospect. (a) Weather noise scale 1.5× crashed accuracy to ~69% because conservative weather vehicles sit at ~28% on the formula, only 2 percentage points below the C/N category boundary. Reverted to 1.0×. (b) Global SIGMA increase to (0.8, 1.5, 0.4) crashed urban (75.9%) and weather (64.9%); the asymmetry across scenarios required a targeted highway-only noise scale rather than a uniform global increase.

## **Final report updates**

Three sections of the final report were updated to reflect the May 2–9 work. Methods now describes both assessor classes and the noise model. Experimental Setup adds the σ values, per-scenario thresholds, and the IDM/MOBIL configuration. Results adds the final agent-vs-ground-truth accuracies as a closing validation table. Limitations section flags the urban accuracy below the original 90–95% target, with the explanation. Final report is now 43 pages.

## **Plan for the final stretch**

Submission goes up Sunday May 10 at 9 AM as agreed with Prof. Daher. Wednesday May 13 final presentation slot remains booked. The deck has been updated with one new slide covering the noise-model validation and the comparison chart, slotted between the existing limitations slide and the conclusion.

## **Updated deliverables checklist**

- Notebooks (this document and the daily research notebook) — uploaded to the VIP team's MS Teams space.

- Peer reviews — completed and emailed to Prof. Joseph Zeaiter directly per his instructions.

- Final report (v2 with noise-model validation) — 43 pages, scheduled for upload Sunday May 10.

- Final presentation deck — updated, 15-minute slot booked for Wednesday May 13.

- Codebase — pushed to the lab GitHub with the term2-final-v2-with-noise-validation tag, README updated to document the two assessor classes and the sensor noise model.

- Datasets — driving_behaviors_dataset.csv (legacy), demo_data.csv (midterm), the SUMO scenario telemetry files generated by the three new scenario runners, sumo_comparison.png, and a 5-minute bag recording demonstrating the full ROS2 stack — all uploaded to the Teams space.

## **Closing note**

The work this term split roughly into three phases: agent-side machine learning (weeks 1–7), systems and scale work (weeks 8–14), and the post-deadline integration push (May 2–9) that produced the agent-vs-ground-truth validation against a sensor-noise model. The first phase produced the gate-head DynamicWeightAgent and the curiosity-augmented Q-learner; the second produced the SUMO shockwave finding and the Social Latency mitigation; the third produced the validation that the closed-form AI score holds up under realistic sensor noise on independent code paths. The pipeline is what makes the project meaningful: a closed-form interpretable Aggressiveness Index that feeds an RL fine-tuning loop, that produces a regime-aware predictor, that publishes on a ROS2 topic, that has been validated against ground truth at 81–91% agreement under realistic noise. Interpretability survives the whole stack.

Plan for next term: CARLA integration, multi-driver coordination experiments, on-vehicle ROS2 deployment, and a longer noise-model exploration (correlated time-domain noise per sensor instead of the i.i.d. Gaussian used here).

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
