# **Weekly Report — Week 7**

**Period: **Tuesday Mar 3 – Monday Mar 9, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Run the long-horizon stability check Prof. Daher requested: 5000 episodes with curiosity on, the agent's predicted weights tracked over time. Pass criterion: high-visitation regimes do not drift away from target weights by more than 0.05 L1. In parallel, install SUMO on the lab machine, get the hello-world scenario running, and write a thin Python wrapper around the TraCI API.

## **Work completed**

Launched the 5000-episode stability run on a tmux session, taking ~6 hours of CPU time. Logged predicted weights every 10 episodes.

Installed Eclipse SUMO 1.20 on the lab machine via apt (`sumo`, `sumo-tools`, `sumo-doc`). Verified the GUI works on a 2-lane sample net.

Wrote env/sumo_wrapper.py exposing the same interface as the highway-env wrapper: reset(), step(), close(), get_vehicles_telemetry(). Used traci.vehicle.getLanePosition() as the lane-internal coordinate for proximity computation.

First end-to-end SUMO run: 3-lane, 5 km motorway scenario, Poisson vehicle spawning at 0.4/sec/lane, baseline speed 100 km/h. 60 seconds of sim time produced 8780 telemetry samples. Mean AI score 18.4, 95th percentile 41.2 — same range as our highway-env runs.

Ran the gate-head agent on these SUMO samples in eval mode (no learning, just inference). Predicted weights cluster at (0.30, 0.11, 0.43, 0.16), within 0.02 L1 of the highway target. The gate's mixture: ~85% highway, ~10% urban, ~5% weather. The agent recognizes SUMO highway as a highway without retraining.

## **Key results**

Stability: highway weights drifted from (0.30, 0.10, 0.45, 0.15) at episode 1 to (0.31, 0.10, 0.45, 0.14) at episode 5000 — drift of 0.02 L1, well within the 0.05 budget. Urban and weather drifted by 0.04 and 0.03 L1 respectively. All three regimes pass the criterion.

Highway shows a slow, monotonic 0.01-magnitude drift toward more weight on speed — same pattern observed in week 4's embedding fine-tune experiment. Suggests the agent finds slightly better weights than the hand-picked targets. Worth a sentence in the final report; not pursuing further this term.

First SUMO end-to-end run is a green build.

## **Plan for next week**

Heterogeneous SUMO scenario: 5% aggressive vehicles (minGap=0.5, accel=4.0, decel=4.5, impatience=1.0), 95% normal vehicles. Run for 5 minutes of simulated time. Compute per-vehicle mean AI score. Confirm aggressive vehicles score higher; check whether neighboring normal vehicles score higher than reference normals from an aggressive-free run (the shockwave hypothesis from the midterm's future-work section).

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
