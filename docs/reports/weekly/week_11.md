# **Weekly Report — Week 11**

**Period: **Tuesday Mar 31 – Monday Apr 6, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Implement the friction-aware regime that the midterm flagged as future work. Add slip-ratio telemetry to the SUMO wrapper, set up a low-friction scenario, verify the gate's weather mixture rises in response to elevated slip, tune the weather-target weights for low friction, and produce the AI_Weather composite formula.

## **Work completed**

Derived slip ratio from kinematic observables. SUMO doesn't expose slip directly, so used slip ≈ |v_actual − v_intended| / v_actual, where v_intended is the IDM equilibrium speed for the current edge density and v_actual is reported by TraCI. On μ=0.9 edges, slip is essentially zero (~0.01 noise floor); on μ=0.4 edges, slip rises to 0.15–0.25 during accel events.

Built a low-friction SUMO scenario: same 3-lane motorway, μ=0.4 across all edges. 5 simulated minutes, ~600 vehicles. Mean slip 0.12, mean speed dropped from 96 km/h (high-friction reference) to 78 km/h.

Added slip ratio as a 5th context feature (was 4-dim, now 5-dim). Ran the gate-head agent on low-friction telemetry in eval mode.

Per-channel correlation analysis with slip ratio: w_accel correlates 0.71 (Pearson), w_wave 0.43, w_prox 0.18, w_speed 0.04. Adjusted the weather target from the original literature-inspired (0.20, 0.30, 0.20, 0.30) to (0.10, 0.40, 0.15, 0.35) — more emphasis on accel and wave, less on prox. Re-trained the gate against the updated target (200 epochs, loss converges to 0.06). Saved as model/gate_v2.pt.

Considered an AI_Weather composite formula bolted on top of AI: AI_Weather = AI · (1 + γ · slip), with γ tuned. After analysis, kept the regime-aware gate as the only mechanism — adding a multiplier on top of AI is a hack over the cleaner architectural choice already in place. Documented the trade-off.

## **Key results**

On the low-friction scenario, the gate's mixture rises to ~50% weather (from ~5% in high-friction reference). AI_Weather population mean: 41.7, vs. 26.3 in high-friction reference.

Bonus result: re-ran the heterogeneous 5%-aggressive scenario in low-friction. Population AI = 49.3; reference low-friction normal = 41.7. Shockwave bump in low-friction = +7.6, slightly higher than the high-friction +6.8. Aggressive drivers are more impactful on slippery roads — kinematic perturbations have higher consequences per unit input on low-grip surfaces.

## **Plan for next week**

ROS2 wrapper. Build a minimal ROS2 node that subscribes to a /telemetry topic, runs the gate-head agent, publishes /env/aggressiveness_index. Use rclpy. Bag-record a 5-minute SUMO simulation pumping telemetry into the topic, replay with rosbag to confirm reproducibility.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
