# **Weekly Report — Week 13**

**Period: **Tuesday Apr 14 – Monday Apr 20, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Two parallel tracks. (1) Add the /env/aggressiveness_summary topic publishing at 1 Hz with population-level aggregates: mean_ai, max_ai, n_affected_neighbors, aggressive_fraction. (2) Draft Sections 1–3 of the final report: introduction, related work, methods. Target 12 pages combined.

## **Work completed**

Wrote SummaryNode in summary_node.py. On startup creates a 1 Hz timer; maintains an in-memory dict of vehicle_id → (timestamp, ai_score, mixture_weights), evicts entries older than 2 seconds, computes population statistics on each tick, publishes on /env/aggressiveness_summary. Custom AggressivenessSummary.msg with mean_ai, max_ai, n_vehicles, n_affected_neighbors, aggressive_fraction.

Tested with a 50-vehicle fleet on the heterogeneous SUMO scenario. Summary topic publishes at exactly 1 Hz; reported aggregates track the ground-truth aggregates from the SUMO trajectory log when computed offline. n_affected_neighbors fluctuates between 8 and 14 in the 5%-aggressive scenario, consistent with week 8's spatial decomposition.

Drafted Section 1 (Introduction, ~2 pages, 1800 words). Three subsections: practical problem, what's missing in the literature, what this project contributes.

Drafted Section 2 (Related Work, ~3 pages). Three threads: classical behavior-classification, deep-learning behavior-classification, interpretable / hybrid approaches. Re-read Zotero references during drafting to avoid mis-citing.

Drafted Section 3 (Methods, parts 3.1–3.5, ~6 pages). 3.1 closed-form AI formulation. 3.2 normalization. 3.3 DynamicWeightAgent architecture (both heads, with a TikZ-rendered architecture diagram). 3.4 count-based curiosity bonus with the discretization and β-tuning derivation. 3.5 Social Latency penalty with the rolling-baseline definition.

## **Key results**

Summary topic publishing at 1 Hz, message size ~32 bytes; aggregates match SUMO ground truth.

Sections 1–3 drafted to ~11 pages.

## **Issues encountered**

During the Monday sync, Prof. Daher flagged that the abstract over-claimed on the SUMO multi-driver experiments. Edited live during the meeting from 184 to ~150 words with conditional language ("in simulation", "in our heterogeneous-driver experiments").

## **Plan for next week**

Sections 4–8 of the final report. Section 4 is experimental setup (SUMO scenarios, hyperparameters, evaluation protocol). Section 5 is the results — shockwave, dose-response, Social Latency mitigation, friction-aware AI_Weather. Sections 6 (discussion), 7 (limitations + future work), 8 (conclusion). Plus an end-to-end smoke test of the full ROS2 + SUMO + agent stack.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
