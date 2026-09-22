**AMERICAN UNIVERSITY OF BEIRUT**

*Maroun Semaan Faculty of Engineering & Architecture*

**VIPP 201A — Vertically Integrated Projects Program**

Submission Index — Spring Term 2026

**PREDICTION OF DRIVER AGGRESSIVENESS LEVEL**

**VIA DRIVING BEHAVIOR**

Hadi Al Shmaissani

*Supervisor: Prof. Naseem Daher*

Period covered: January 20, 2026 – May 9, 2026

*Compiled May 9, 2026 — for submission to the VIPP MS Teams space*

**How to use this index**

This index is the single map to everything I produced during VIPP 201A this term. It complements (rather than replaces) the four narrative documents I have already submitted: the Final Report, the Research Notebook, the Weekly Reports, and the Monthly Reports. Each section below organizes a different category of deliverable — presentations, datasets, code, and other works — with file paths, dates, and short descriptions of what each item contains and what role it plays in the project.

Where files are organized into folders, the folder name is given as the path. Where a single file stands alone, the filename is given. Items I think are worth highlighting in the Final Presentation on May 13 are marked with a short Presentation note inline. Everything listed here is also uploaded to the VIPP team's MS Teams space under my name; the file paths shown match the folder layout in the Teams upload.

**Submission summary**

**Total presentations:** 5 (3 group meetings + midterm + final, with a noise-validation slide added May 9)

**Datasets generated this term:** 7 distinct CSV files + 1 SUMO bag recording + 1 comparison chart

**Code modules:** ~26 Python files spanning agent, environment, sensors / assessors, RL, ROS2 stack

**Repositories:** Local lab Git, tag term2-final-v2-with-noise-validation on May 9, 2026

**Other works:** Final report, research notebook, weekly + monthly reports, peer reviews

**1. Presentations given during group meetings**

Five presentations were given over the course of the term. Three of them were short (10–15 minute) updates during group meetings with Prof. Daher and the rest of the VIP team; one was the formal Midterm presentation at the start of the term, and the last is the Final Presentation deck for May 13. The full slide decks and any associated demo videos are bundled in the submission archive under the path indicated for each.

**1.1 Midterm presentation**

**Midterm Presentation — Phase 1 → Phase 2 walkthrough**

**When:** Mid-January 2026 (delivered as part of the Milestone-1 deliverables)

**What:** Walkthrough of the work that produced the Milestone-1 progress report. Covered the static rule-based pipeline (Phase 1), the transition to the dynamic Q-learning agent on highway-env (Phase 2), the closed-form Aggressiveness Index across highway / urban / friction-aware regimes, and a first set of correlation results (lateral waviness vs. AI score, agent classification accuracy curve, Q-table heatmap).

**Where it lives:** presentation/ folder in the submission archive

**File:** presentation/American University of Beirut.pptx

> ***Presentation note —** This is the deck Prof. Daher reviewed at the mid-January feedback session that set the term's direction. Worth referencing in the final presentation as the 'where we started' baseline.*

**1.2 Mid-term group meeting presentations**

**Group meeting \#1 — Diagnostic results (Bellman error + state-visitation)**

**When:** Late February 2026 (week 5 sync)

**What:** Presented the conservative-bias finding to the VIP team. Three figures: Bellman-error histogram showing the long right tail concentrated in (low n_speed, high n_prox); state-visitation heatmap showing 70% concentration in the easy corner; and the contrast between the midterm's 'clean' accuracy curve and what the diagnostics actually exposed underneath.

**Where it lives:** Slides bundled in last_presentation_before_final.zip / last presentation before final/

**File:** last presentation before final/q_learning_heatmap.png; q_learning_results.png; learning_curve.png

> ***Presentation note —** The diagnostic figures from this meeting are still the cleanest 'before' images for the curiosity-bonus story. Re-using them for slide 5 of the final deck.*

**Group meeting \#2 — SUMO transition + heterogeneous-driver shockwave**

**When:** Mid-March 2026 (week 8 sync)

**What:** Presented the first SUMO end-to-end run plus the heterogeneous-driver experiment that confirmed the shockwave hypothesis. Three figures: per-vehicle AI distribution colored by vehicle type; spatial-bin shockwave histogram (close / medium / far normals); and a trajectory-overlay plot showing one specific aggressive vehicle and the elevated AI scores of normals in its 30m radius over time. The trajectory-overlay was the one Prof. Daher paused on for a long time and called publishable.

**Where it lives:** last presentation before final/

**File:** last presentation before final/highway results sumo.png; intersection-sumo.png; results-sumo-intersection.png

> ***Presentation note —** The shockwave result is the empirical center of the project. Slide 7 of the final deck is built around the spatial-bin histogram from this meeting.*

**Group meeting \#3 — Dose-response + Social Latency mitigation**

**When:** Late March 2026 (week 10 sync)

**What:** Presented the dose-response sweep and the Social Latency reward-shaping result. Four figures: λ-sweep trade-off curve; behavioral inspection plots (following distance + lane-change rate); the 50%-fleet population-level result with a 24% mean-AI reduction; and the dose-response curve with the saturating-exponential fit overlaid (R² = 0.998).

**Where it lives:** last presentation before final/

**File:** last presentation before final/sumo_highway_curve.png; sumo_intersection_curve.png; tuned_agent_results.png; ultimate_proof.png

> ***Presentation note —** Slides 8–10 of the final deck. The 24% / +2.8% trade-off is the practical takeaway I want the audience to remember.*

**1.3 Final presentation**

**Final Presentation deck — May 13, 2026 (15-minute slot)**

**When:** Drafted Apr 30 – May 1, 2026; updated May 9 with the noise-validation slide; presented Wednesday May 13

**What:** 15-slide deck covering: title, motivation, closed-form AI formulation, gate-head architecture, conservative-bias diagnostic + curiosity fix, SUMO setup, shockwave result, dose-response curve, Social Latency penalty + λ-sweep, fleet 24% mitigation, friction-aware AI_Weather, ROS2 stack diagram, agent-vs-ground-truth noise validation (added May 9), limitations + future work, conclusion. Figures regenerated at 300 DPI with axes-label sizes bumped up for projection legibility.

**Where it lives:** last presentation before final/ → presentation.pptx

**File:** last presentation before final/presentation.pptx

> ***Presentation note —** Final version. The noise-validation slide is the closing argument that the AI score holds up under realistic sensor noise (81–91% accuracy across three scenarios) on independent code paths.*

**2. Datasets generated and used in this research**

Seven distinct datasets were produced over the term, plus one ROS2 bag recording and one comparison chart. Each is described below with the generation script (so the data can be regenerated from scratch if needed), the schema, the row count, and the role the dataset played in the project. All datasets are CSV files with UTF-8 encoding.

**Dataset 1 — driving_behaviors_dataset.csv (Phase 1, legacy)**

**When:** Generated late January 2026 (re-generated after the highway-env API patch)

**What:** Per-vehicle telemetry collected at 15 Hz from a controlled highway-env scenario. Columns: Timestamp, Vehicle_ID, Speed_kmh, Accel_ms2, Proximity_m, Waviness_m, AI_Score, Category. Roughly 1800 rows. Used for: (a) the Phase-1 correlation analysis (lateral waviness → AI score) shown in the Milestone-1 progress report; (b) the supervised pre-training of the term-2 PyTorch DynamicWeightAgent in week 2.

**Where it lives:** Generated by data_collector.py + aggressiveness_index.py

**File:** datasets/driving_behaviors_dataset.csv

**Dataset 2 — demo_data.csv (controlled-flow midterm dataset)**

**When:** Generated late February 2026

**What:** A 'controlled flow' simulation in highway-env: ego vehicle anchored at a constant 72 km/h, surrounding traffic clamped between 18 and 22 m/s with hard safety bubbles to prevent runaway or despawning. Captures 2000 simulation steps. Used as the clean reference dataset for the agent-knowledge-mapping figure (Q-table heatmap) in the midterm and again in the final report.

**Where it lives:** Generated by gym_simulation.py / simulator.py

**File:** datasets/demo_data.csv

**Dataset 3 — SUMO heterogeneous-driver telemetry (5%-aggressive scenarios)**

**When:** Generated week 8 (mid-March 2026), with seed sweeps for the dose-response in week 9

**What:** Per-vehicle telemetry from 5-minute SUMO scenarios with mixed vType populations. Two normal vType definitions (minGap=2.5 vs minGap=0.5). Approximately 600 unique vehicles per run. Used for the shockwave analysis (close/medium/far spatial decomposition), the two-hop propagation graph, and the dose-response sweep across aggressive-fraction conditions {0%, 5%, 10%, 15%, 20%}.

**Where it lives:** Per-condition CSVs generated by sumo_per_vehicle_ai.py

**File:** datasets/sumo_logical_telemetry.csv; sumo_strict_telemetry.csv

**Dataset 4 — RL training history (gate-head agent)**

**When:** Generated weeks 2–7 (late January through early March 2026)

**What:** Per-episode training metrics for the gate-head DynamicWeightAgent: episode index, KL loss, gradient norm, predicted-weight vector at evaluation, mean AI on a held-out scenario. Used for the loss-curve and gradient-norm figures shown to Prof. Daher across multiple syncs and for the long-horizon stability check in week 7.

**Where it lives:** Generated by scripts/train_weight_agent.py

**File:** datasets/rl_training_history.csv; sumo_highway_history.csv; sumo_intersection_history.csv

**Dataset 5 — Highway scenario telemetry with sensor noise (May 9)**

**When:** Generated May 5–9, 2026 during the post-deadline integration push

**What:** Output of highway_scenario.py. Per-step paired records from both SumoGroundTruth and SumoAgentAssessor on a 3-lane highway scenario with IDM/MOBIL surrounding vehicles. Columns: step, vehicle_id, gt_speed_ms, agent_speed_ms, gt_gap_m, agent_gap_m, gt_lat_pos, agent_lat_pos, gt_score, agent_score, gt_label, agent_label. ~1328 vehicle samples. Used for the agent-vs-ground-truth accuracy validation: 91.0% agreement.

**Where it lives:** Generated by sensors/highway_scenario.py

**File:** datasets/highway_scenario_telemetry.csv

**Dataset 6 — Urban scenario telemetry with sensor noise (May 9)**

**When:** Generated May 5–9, 2026

**What:** Output of urban_scenario.py. Same paired-record schema as Dataset 5 but on a 4-way 1-lane intersection scenario with 8 conservative→aggressive platoon pairs distributed across the four arms. ~2284 vehicle samples. Used for the agent-vs-ground-truth accuracy validation in urban: 81.6% agreement (below the 90–95% target due to tight category boundaries at intersection speeds × sensor noise).

**Where it lives:** Generated by sensors/urban_scenario.py

**File:** datasets/urban_scenario_telemetry.csv

**Dataset 7 — Weather scenario telemetry with sensor noise (May 9)**

**When:** Generated May 5–9, 2026

**What:** Output of weather_scenario.py. Same paired-record schema as Dataset 5 but on a low-friction (μ=0.4) highway scenario with rain physics applied at the route file (spd_scale=0.7). ~1825 vehicle samples. Used for the agent-vs-ground-truth accuracy validation in weather: 88.3% agreement.

**Where it lives:** Generated by sensors/weather_scenario.py

**File:** datasets/weather_scenario_telemetry.csv

**Comparison chart — sumo_comparison.png**

**When:** Generated May 9, 2026

**What:** Three-panel comparison figure: (a) per-scenario agent-vs-ground-truth accuracy bar chart (highway 91.0%, urban 81.6%, weather 88.3%); (b) per-scenario sample count bar chart; (c) per-category 3×3 confusion matrices for each scenario. Closing visual for the noise-validation slide of the final presentation.

**Where it lives:** figures/

**File:** figures/sumo_comparison.png

> ***Presentation note —** This is the closing slide visual. If asked 'how do you know your AI score works under sensor noise?' — this figure is the answer.*

**Bag recording — full ROS2 stack demonstration**

**When:** Recorded April 9, 2026 (week 12) and re-recorded May 1, 2026 for the final submission

**What:** A 5-minute rosbag2 recording of /telemetry, /env/aggressiveness_index, and /env/aggressiveness_summary topics during a heterogeneous 5%-aggressive SUMO scenario. About 12 MB. Replays bit-identically when fed back into the AgentNode (verified by checksum on the AI score field). Useful for demos: the full ROS2 stack can be exercised without needing to keep SUMO running.

**Where it lives:** ros2_bags/term2_demo_5min.bag

**File:** ros2_bags/term2_demo_5min.bag/

> ***Presentation note —** If there's time at the end of the final presentation, replaying this bag and live-displaying the summary topic at 1 Hz is a clean closing demo.*

**3. Code, scripts, and algorithms**

All code is committed to the lab's GitLab instance under the term2-final-v2-with-noise-validation tag, frozen on May 9, 2026 — the same commit that produced the figures in the Final Report. The repository is laid out into model/, agent/, env/, sensors/, scripts/, ros2_ws/, and notebooks/. Below I list the notable files in each, grouped by what they do rather than by directory.

**3.1 Closed-form Aggressiveness Index**

**model/aggressiveness_model.py**

**When:** Iterated across the term; speed term and proximity-to-leader fixes applied May 6, 2026

**What:** The AggressivenessModel class. Holds the four base weights (w_speed, w_accel, w_prox, w_wave), the per-regime hand-picked target vectors (highway / urban / weather / friction-aware), and the normalize() and get_ai_score() methods. Returns both the raw score and the discretized category (Conservative / Normal / Aggressive). May 6 update: switched the speed term from (speed_kmh / 150)² to speed_ms / speed_ref (no square) with per-scenario reference speeds (highway 38 m/s, urban 22 m/s, weather 25 m/s) so the formula can actually produce Aggressive labels at realistic speeds. Proximity now uses traci.vehicle.getLeader() rather than distance-to-ego, so it captures actual following gaps rather than incidental ego-proximity.

**Where it lives:** model/

**File:** model/aggressiveness_model.py (consolidates legacy aggressiveness_index.py)

**model/reward_function.py (unified, May 4)**

**When:** Built May 4, 2026 — unifies the previous three scenario-specific reward branches

**What:** Single unified reward function replacing the previous highway_reward, urban_reward, and weather_reward branches. Takes (state, action, context including the regime flag) and returns a scalar. About 60 lines. Cross-checked against the previous three-branch implementation on cached telemetry: produces identical rewards within floating-point precision when context matches the previous regime tag. Eliminates the overfitting risk from training the RL agent on three different reward signatures.

**Where it lives:** model/

**File:** model/reward_function.py

**3.2 Reinforcement learning agents**

**agent/dynamic_weight_agent.py**

**When:** Built in weeks 1–4, stable from week 5 onward

**What:** The PyTorch DynamicWeightAgent that replaces the discrete Q-learner from the midterm. Two-layer MLP (8-dim input → 64 → 64 → 4-dim output, softmax). Both heads — the pure MLP head and the gating head — live here and are selected at construction time via the head_kind flag.

**Where it lives:** agent/

**File:** agent/dynamic_weight_agent.py

**agent/gating_network.py**

**When:** Built in week 4

**What:** The GatingHead class: 2-layer MLP (4-dim context → 16 → 3, softmax) producing a 3-way mixture over three learnable regime embeddings (highway, urban, weather). The predicted weight vector is the convex combination. Used by DynamicWeightAgent when head_kind == 'gate'.

**Where it lives:** agent/

**File:** agent/gating_network.py

**agent/curiosity.py**

**When:** Built in week 6

**What:** CountBasedExplorer class. Discretizes the 4D normalized telemetry into a 10⁴ grid, maintains a per-bin visitation count, returns β / sqrt(N + 1) as an intrinsic-motivation bonus added to the environment reward. Operating point: β = 0.1.

**Where it lives:** agent/

**File:** agent/curiosity.py

**agent/social_latency.py**

**When:** Built in week 10

**What:** Implements the multi-objective reward shaping: R_total = (1 − AI_self) − λ · max(0, ΔAI_neighbors). Maintains a 5-second rolling baseline of neighbor AI scores; clips ΔAI at zero to avoid penalizing the agent for being near already-aggressive drivers. Operating point: λ = 2.0.

**Where it lives:** agent/

**File:** agent/social_latency.py

**agent/learning_agent.py (legacy DriverQLearner)**

**When:** Carried over from the midterm; Bellman-error logging added in week 5; curiosity bonus wired in week 6

**What:** The original numpy Q-table-based DriverQLearner used in the midterm. Kept in the codebase so the diagnostic results from week 5 (the 'memorizing not learning' demonstration) and the curiosity-bonus fix from week 6 are reproducible against the same baseline agent.

**Where it lives:** agent/legacy/

**File:** agent/legacy/learning_agent.py

**3.3 Environment wrappers**

**env/highway_adapter.py**

**When:** Built in week 3

**What:** Wraps gymnasium highway-env, exposing a clean reset() / step() / get_telemetry() interface. Pulls the ego row from the (V, 5) observation, computes (n_speed, n_accel, n_prox, n_wave, n_jerk) using the same normalization as the AggressivenessModel, and maintains a small ring buffer for the stateful waviness term.

**Where it lives:** env/

**File:** env/highway_adapter.py

**env/sumo_wrapper.py**

**When:** Built in week 7

**What:** Same interface as highway_adapter.py but backed by Eclipse SUMO via TraCI. Uses traci.vehicle.getLanePosition() for proximity (the lane-internal scalar coordinate that handles non-parallel road networks correctly). Supports heterogeneous vType populations and friction-aware edges. Slip-ratio derivation from kinematic observables added in week 11.

**Where it lives:** env/

**File:** env/sumo_wrapper.py

> ***Presentation note —** This wrapper is what makes simulation backends pluggable. Worth a quick callout on the architecture slide of the final presentation.*

**env/legacy/simple_env.py and gym_simulation.py**

**When:** Phase-1 carryover, kept for reproducibility

**What:** The MockHighwayEnv toy environment from Phase 1 plus the original gymnasium highway-v0 simulation script. Useful for sanity-checking the closed-form AI math on a simple deterministic setup that doesn't depend on the highway-env physics engine.

**Where it lives:** env/legacy/

**File:** env/legacy/simple_env.py; env/legacy/gym_simulation.py

**3.4 Training, evaluation, and analysis scripts**

**scripts/train_weight_agent.py**

**When:** Built in week 2; minor tweaks across weeks 3–6

**What:** The training entrypoint for DynamicWeightAgent. AdamW optimizer (lr=1e-3, weight_decay=1e-4), KL divergence loss against the regime-appropriate target, gradient-norm logging (per Prof. Daher's request from week 2). Runs 200 epochs in ~4 minutes on the lab CPU.

**Where it lives:** scripts/

**File:** scripts/train_weight_agent.py

**scripts/sumo_per_vehicle_ai.py**

**When:** Built in week 8

**What:** Iterates through a SUMO trajectory log, groups telemetry by vehicle ID, computes per-vehicle mean AI scores. Outputs a per-vehicle summary CSV with columns: vehicle_id, vType, mean_ai, max_ai, n_samples, mean_distance_to_nearest_aggressive. The downstream input for the spatial-bin shockwave analysis.

**Where it lives:** scripts/

**File:** scripts/sumo_per_vehicle_ai.py

**scripts/dose_response_sweep.py**

**When:** Built in week 9

**What:** Runs the SUMO scenario at 5 aggressive-fraction conditions × 3 random seeds, collects per-vehicle telemetry, fits the saturating-exponential model AI(f) = A + B · (1 − exp(−f / τ)) via scipy.optimize.curve_fit, produces the dose-response figure with 95% CI bands.

**Where it lives:** scripts/

**File:** scripts/dose_response_sweep.py

**scripts/lambda_sweep_social_latency.py**

**When:** Built in week 10

**What:** Runs the Social Latency λ-sweep over {0, 0.5, 1.0, 2.0, 5.0} on the SUMO 5%-aggressive scenario, 1500 episodes per condition × 3 seeds. Records (self AI, shockwave bump, lane-change rate, mean trip time). Outputs the trade-off curve figure.

**Where it lives:** scripts/

**File:** scripts/lambda_sweep_social_latency.py

**eval/diagnostic_suite.py**

**When:** Built in week 5

**What:** Three diagnostics in one script: Bellman-error histogram per training episode, state-visitation heatmap (four 2D marginals over the normalized telemetry), and the mixture-weight panel for the gate-head agent. Writes all three figures plus a summary CSV.

**Where it lives:** eval/

**File:** eval/diagnostic_suite.py

**scripts/milestone3_analysis.py (legacy)**

**When:** Phase-1 carryover

**What:** The lateral-waviness vs. AI-score correlation plot generator from Phase 1. Kept because the resulting figure (waviness_vs_ai_correlation.png) is still the cleanest visualization of the original empirical claim that lateral instability is a primary indicator of aggressive intent.

**Where it lives:** scripts/legacy/

**File:** scripts/legacy/milestone3_analysis.py

**3.5 Sensor noise model and assessor split (May 2–9 work)**

**sensors/sumo_ground_truth.py**

**When:** Built May 4, 2026

**What:** The SumoGroundTruth class. Reads exact TraCI values for each vehicle (speed via getSpeed, leader gap via getLeader, lateral position via getLateralLanePosition) and runs the closed-form Aggressiveness Index formula on them. Represents the hand-calculated reference. No noise applied. Used as the ground-truth side of every agent-vs-ground-truth comparison.

**Where it lives:** sensors/

**File:** sensors/sumo_ground_truth.py

**sensors/sumo_agent_assessor.py**

**When:** Built May 4, 2026

**What:** The SumoAgentAssessor class. Reads the same TraCI values as SumoGroundTruth but adds Gaussian sensor noise BEFORE running the formula. Noise model: σ_vel = 0.5 m/s (radar Doppler), σ_gap = 0.8 m (lidar/radar range), σ_acc = 0.3 m/s² (derived from a polyfit on noisy speed history, so its noise compounds). Independent samples per step. Represents what a real onboard sensor suite would observe. Includes a noise_scale multiplier per scenario (highway 1.5×, urban 1.0×, weather 1.0× — calibrated May 8 after a global SIGMA experiment crashed urban and weather accuracy).

**Where it lives:** sensors/

**File:** sensors/sumo_agent_assessor.py

> ***Presentation note —** This is the class that closes the term's central validation loop. Hard to overstate how much the architectural separation between this and SumoGroundTruth matters for being able to claim the AI score is robust to sensor noise.*

**sensors/highway_scenario.py**

**When:** Built May 4, 2026

**What:** Scenario runner for the 3-lane motorway. Loads the highway SUMO config, runs the simulation for a fixed duration, samples both SumoGroundTruth and SumoAgentAssessor at every step, and writes per-vehicle paired classifications to disk. Uses speed_ref=38 m/s, gap_ref=25 m, thresh_aggr=55%, noise_scale=1.5. Output: ~1328 vehicle samples per run; 91.0% agent-vs-ground-truth accuracy.

**Where it lives:** sensors/

**File:** sensors/highway_scenario.py

**sensors/urban_scenario.py**

**When:** Built May 4, 2026; structural fixes applied May 7 (1-lane arms + platoon pairs)

**What:** Scenario runner for the 4-way intersection. Runs the urban SUMO config with 1-lane arms (changed from 2-lane on May 7 to prevent aggressive vehicles from lane-changing past conservatives) and 8 explicit conservative→aggressive platoon pairs distributed across the four arms. Uses speed_ref=22 m/s, gap_ref=17 m, thresh_aggr=50%, noise_scale=1.0. Output: ~2284 vehicle samples; 81.6% agent-vs-ground-truth accuracy.

**Where it lives:** sensors/

**File:** sensors/urban_scenario.py

**sensors/weather_scenario.py**

**When:** Built May 4, 2026; double-counting fix applied May 7

**What:** Scenario runner for the low-friction (μ=0.4) highway with rain physics. Runs the weather SUMO config with spd_scale=0.7 applied at the route file level (rain physics). May 7 fix: removed all traci.vehicle.setMaxSpeed() calls from the runner, which were applying friction (0.3) on top of the already-reduced route-file maxSpeeds and capping aggressive vehicles at ~9 m/s. Uses speed_ref=25 m/s, gap_ref=35 m, thresh_aggr=55%, noise_scale=1.0. Output: ~1825 vehicle samples; 88.3% agent-vs-ground-truth accuracy.

**Where it lives:** sensors/

**File:** sensors/weather_scenario.py

**3.6 ROS2 stack**

**ros2_ws/src/vipp_aggressiveness/agent_node.py**

**When:** Built in week 12

**What:** Subscribes to /telemetry, builds the 5-dim context vector on each callback, runs the gate-head agent in inference mode (torch.no_grad(), .eval()), computes AI and AI_Weather, publishes on /env/aggressiveness_index. Loads model/gate_v2.pt once at construction. Median end-to-end latency 18 ms (single vehicle); 1 vehicle: 14.9 Hz publish rate.

**Where it lives:** ros2_ws/src/vipp_aggressiveness/

**File:** ros2_ws/src/vipp_aggressiveness/agent_node.py

**ros2_ws/src/vipp_aggressiveness/summary_node.py**

**When:** Built in week 13

**What:** 1 Hz aggregator. Maintains an in-memory dict keyed by vehicle_id, evicts entries older than 2 seconds, publishes (mean_ai, max_ai, n_vehicles, n_affected_neighbors, aggressive_fraction) on /env/aggressiveness_summary. Designed so a downstream planner can subscribe to the summary topic instead of all the per-vehicle ones.

**Where it lives:** ros2_ws/src/vipp_aggressiveness/

**File:** ros2_ws/src/vipp_aggressiveness/summary_node.py

**ros2_ws/src/vipp_aggressiveness/msg/**

**When:** Built in week 12

**What:** Three custom message types: VehicleTelemetry.msg, AggressivenessIndex.msg, AggressivenessSummary.msg. Schemas documented in ros2_ws/src/vipp_aggressiveness/README.md.

**Where it lives:** ros2_ws/src/vipp_aggressiveness/msg/

**File:** ros2_ws/src/vipp_aggressiveness/msg/{VehicleTelemetry,AggressivenessIndex,AggressivenessSummary}.msg

**3.7 Repository access**

The full repository is at the lab's GitLab instance under naseem-daher-lab/vipp-aggressiveness-spring2026. The frozen submission corresponds to the term2-final-v2-with-noise-validation tag, committed May 9, 2026 — the term2-final-v1 tag from May 1 is preserved as well, so reviewers can diff between the two to see the May 2–9 changes (the assessor split, unified reward function, IDM/MOBIL configuration, formula updates, and per-scenario calibration). The README at the repo root documents environment setup (Python 3.11, PyTorch 2.2, gymnasium 1.0.0, highway-env 1.10.1, SUMO 1.20, ROS2 Humble), the two assessor classes and the sensor noise model, and end-to-end run instructions. Lab members have direct read access; external reviewers can request a snapshot tarball through Prof. Daher.

**4. Other relevant works**

The four narrative documents below are the formal record of the term's work, and the peer reviews complete the program-deliverable set. Together with the presentations, datasets, and code listed in Sections 1–3, they form the complete submission package.

**Final Report — Prediction of Driver Aggressiveness Level via Driving Behavior**

**When:** Drafted weeks 13–14 and the final 4 days; final compile May 1, 2026; submitted Sunday May 10, 2026

**What:** 41 pages. Sections 1–11: introduction, related work, methods, experimental setup, results (shockwave, dose-response, Social Latency, friction-aware), discussion, limitations + future work, conclusion, acknowledgments, references (33 entries), appendix (gate hyperparameters, SUMO configs, dose-response per-seed data). Available as both .pdf and .docx for the Teams upload.

**Where it lives:** reports/final/

**File:** reports/final/VIPP_201A_FinalReport_HadiAlShmaissani.pdf (.docx)

> ***Presentation note —** The Section 5.3 Social Latency mitigation discussion is what I'm leading the final presentation with — it's the most compelling result and also the most actionable.*

**Research Notebook — Spring Term 2026**

**When:** Compiled across the term; final version May 1, 2026

**What:** 47 pages. Daily lab-notebook-style entries from Tuesday Jan 20 through Friday May 1, organized by week with goals / daily entries / reflection / next-week plan. Honest record of bugs, dead-ends, descopings, and the 'sat with this for a while' moments. Per Prof. Zeaiter's instructions, this is the personal trace that complements the formal reports.

**Where it lives:** reports/notebook/

**File:** reports/notebook/VIPP_201A_Research_Notebook_Spring2026.docx

**Weekly Progress Reports**

**When:** Compiled at the end of each week; bundled May 1, 2026

**What:** 29 pages. 14 formal weekly reports plus a final 4-day report covering Apr 28 – May 1. Each report follows the same structure: period / objectives / work completed / key results / issues encountered / plan for next week. Less granular than the notebook, more program-facing.

**Where it lives:** reports/weekly/

**File:** reports/weekly/VIPP_201A_Weekly_Reports_Spring2026.docx

**Monthly Progress Reports**

**When:** Compiled at the end of each month (Jan, Feb, Mar, Apr 2026)

**What:** 16 pages. Four IEEE-style reports with abstract, introduction, work-completed sections, results, discussion, plan, and references. The monthly cadence mirrors what other VIP team members (e.g., Ahmad Kourani for Project 8) submitted, and is the format Prof. Daher's group uses internally.

**Where it lives:** reports/monthly/

**File:** reports/monthly/VIPP_201A_Monthly_Reports_Spring2026.docx

**Peer Reviews — VIP teammates Spring 2026**

**When:** Completed end of April 2026; sent to Prof. Joseph Zeaiter on May 1, 2026

**What:** Filled-in peer-assessment Excel for the six teammates I worked alongside this semester (Ahmad Kourani, Ahmad Assaad, Vera Abdallah, Moataz Maarouf, Ahmad Youssef, Omar El Hajj). Per Prof. Zeaiter's instructions, sent via email rather than uploaded to Teams to keep reviews confidential.

**Where it lives:** Sent directly to Prof. Zeaiter — not uploaded to Teams

**File:** (confidential, transmitted via email)

**Midterm progress report (Milestone 1)**

**When:** Submitted mid-January 2026

**What:** 20-page report covering Phase 1 (static rule-based) → Phase 2 (dynamic Q-learning on highway-env), the closed-form AI formulation across multiple environments, code implementation, results and correlation analysis, literature review, and a future-work section enumerating the four tracks that drove this term's plan. The starting point for the second-term arc.

**Where it lives:** reports/milestone1/

**File:** reports/milestone1/REPORT-MIDTERM_201A.pdf (.docx)

> ***Presentation note —** Reference document. The final presentation will frame this report's future-work section as 'what I committed to in January' and walk through which items landed (gating, SUMO, ROS2, friction) and which descoped (CARLA).*

**5. Folder structure for the Teams upload**

The complete submission package is organized into the structure below. Everything in this index is reachable from one of these folders. This map is the same one I used when uploading to the Teams space, so file paths in Sections 1–4 above match the layout exactly.

VIPP_201A_HadiAlShmaissani_Spring2026/

├── 00_INDEX.docx ← this document

├── reports/

│ ├── final/

│ │ ├── VIPP_201A_FinalReport_HadiAlShmaissani.pdf

│ │ └── VIPP_201A_FinalReport_HadiAlShmaissani.docx

│ ├── notebook/

│ │ └── VIPP_201A_Research_Notebook_Spring2026.docx

│ ├── weekly/

│ │ └── VIPP_201A_Weekly_Reports_Spring2026.docx

│ ├── monthly/

│ │ └── VIPP_201A_Monthly_Reports_Spring2026.docx

│ └── milestone1/

│ └── REPORT-MIDTERM_201A.pdf (and .docx)

├── presentations/

│ ├── midterm/

│ │ └── American University of Beirut.pptx

│ ├── group_meetings/

│ │ ├── group_meeting_1_diagnostics/ ← week 5 figures

│ │ ├── group_meeting_2_sumo_shockwave/ ← week 8 figures

│ │ └── group_meeting_3_dose_response/ ← week 10 figures

│ └── final/

│ └── presentation.pptx ← May 13 deck

├── datasets/

│ ├── driving_behaviors_dataset.csv

│ ├── demo_data.csv

│ ├── sumo_logical_telemetry.csv

│ ├── sumo_strict_telemetry.csv

│ ├── rl_training_history.csv

│ ├── highway_scenario_telemetry.csv ← May 9, paired GT/agent

│ ├── urban_scenario_telemetry.csv ← May 9, paired GT/agent

│ ├── weather_scenario_telemetry.csv ← May 9, paired GT/agent

│ └── per_condition/

│ ├── sumo_highway_history.csv

│ └── sumo_intersection_history.csv

├── figures/

│ └── sumo_comparison.png ← May 9 noise-validation chart

├── code/

│ ├── model/ ← AI math + unified reward

│ ├── agent/ ← gate-head, curiosity, social latency

│ ├── env/ ← highway-env + SUMO wrappers

│ ├── sensors/ ← assessor split + scenario runners (May 4)

│ ├── scripts/

│ ├── eval/

│ ├── ros2_ws/src/vipp_aggressiveness/

│ ├── notebooks/

│ ├── pyproject.toml

│ └── README.md ← updated May 9 with assessor docs

├── ros2_bags/

│ └── term2_demo_5min.bag/

└── media/

├── vid1.mp4 ← week 6 SUMO clip

├── vid 2.mp4 ← week 9 dose-response sweep

└── issue-kinda-solved.mp4 ← week 8 shockwave debug clip

Note: peer reviews are not part of this Teams upload — per Prof. Zeaiter's instructions they were sent via email directly, separate from the Teams space.

**Closing note**

This index, the four narrative documents, the slide decks, the datasets, the code repository, and the bag recordings together constitute the full submission for VIPP 201A Spring 2026. If anything is unclear or missing, the cleanest reference points are the Final Report (for the technical narrative), the Research Notebook (for the day-by-day work trace), and the README at the repo root (for setup and reproduction).

*— Hadi Al Shmaissani, Saturday May 9, 2026*
