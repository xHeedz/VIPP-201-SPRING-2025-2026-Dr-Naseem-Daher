# **Weekly Report — Week 12**

**Period: **Tuesday Apr 7 – Monday Apr 13, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Build a ROS2 node that wraps the agent and the AI score: subscribes to a /telemetry topic, runs the gate-head agent on incoming messages, publishes the AI score on /env/aggressiveness_index. Use rclpy (Python ROS2 client). Bag-record a 5-minute SUMO sim, replay to confirm reproducibility. Pass criterion: messages publish at ≥10 Hz with median end-to-end latency under 50 ms.

## **Work completed**

Set up a colcon workspace at ~/ros2_ws and created a vipp_aggressiveness package. Defined two custom message types: VehicleTelemetry.msg (speed, accel, prox, wave, jerk, slip, vehicle_id, stamp) and AggressivenessIndex.msg (ai_score, ai_weather, mixture_weights[3], vehicle_id, stamp).

Implemented AgentNode in agent_node.py. Subscribes to /telemetry, builds the 5-dim context vector, runs the gate-head agent in inference mode (torch.no_grad(), .eval()), computes AI score via AggressivenessModel, publishes on /env/aggressiveness_index. Agent is loaded once at construction time from model/gate_v2.pt.

Wrote a TelemetryPublisherNode that reads a SUMO trajectory log line by line and publishes telemetry messages at the recorded rate. Acts as a deterministic replay source for verifying reproducibility.

Bag-recorded 5 minutes of telemetry (~12 MB), replayed it, and verified output messages on /env/aggressiveness_index are bit-identical (checksum-equal on the AI score field over the full bag). Reproducibility confirmed.

Extended AgentNode to handle multiple vehicle IDs. Single /telemetry topic, internal routing by vehicle_id, per-vehicle ring buffers for the wave/jerk computation, per-vehicle AggressivenessIndex messages keyed by vehicle_id.

## **Key results**

Single-vehicle: median end-to-end latency 18 ms, 99th percentile 41 ms. Topic publishes at 14.9 Hz (matching SUMO's 15 Hz step rate within sub-step jitter). Both within budget.

Multi-vehicle: at 50 vehicles, 99th percentile latency grows to 78 ms. Above the 50 ms budget at high load. Profiled with cProfile; most of the time is in the agent's forward pass (PyTorch overhead at single-sample batches). Documented as a known limitation; can be addressed in future work via small-window batching.

Reproducibility: replay output is bit-identical to original. Unit test added that ingests a 100-message slice and checks output equality.

## **Plan for next week**

Add a /env/aggressiveness_summary topic publishing at 1 Hz with population-level aggregates (mean AI, max AI, neighbor-affected count, fraction above aggressive threshold). Begin drafting the final report — Sections 1–3 (introduction, related work, methods).

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
