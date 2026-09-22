# **Weekly Report — Week 8**

**Period: **Tuesday Mar 10 – Monday Mar 16, 2026

**Student: **Hadi Al Shmaissani

**Project: **Prediction of Driver Aggressiveness Level via Driving Behavior (Project 3)

**Supervisor: **Prof. Naseem Daher

## **Objectives**

Heterogeneous-driver experiment in SUMO. 5% aggressive vehicles, 95% normal, 5 minutes of simulation time. Compute per-vehicle mean AI score and confirm: (a) aggressive vehicles score higher than normals, and (b) normal vehicles spatially adjacent to aggressive ones score higher than reference normals from an aggressive-free run.

## **Work completed**

Defined two vType blocks in the SUMO routes file. Normal: minGap=2.5, accel=2.5, decel=4.5, impatience=0.0, sigma=0.5. Aggressive: minGap=0.5, accel=4.0, decel=4.5, impatience=1.0, sigma=0.7. Probability flag = 0.05 selects aggressive vehicles from the demand distribution. Confirmed visually in SUMO GUI: aggressive vehicles tailgate and lane-change more often.

Wrote scripts/sumo_per_vehicle_ai.py to group telemetry samples by vehicle ID and compute per-vehicle mean AI scores.

Ran the aggressive-free reference scenario (0% aggressive) for the same 5-minute duration to measure baseline normal AI.

Performed spatial decomposition of the shockwave: for each normal vehicle, computed minimum-distance trajectory to any aggressive vehicle, binned normals into close (avg dist < 30m at any point), medium (30–100m), and far (>100m).

Lit check: Kesting & Treiber 2013, Treiber et al. 2000 (IDM), and recent car-following heterogeneity papers. The general finding — aggressive drivers in heterogeneous fleets raise local accelerations and braking events for nearby vehicles — is consistent with our result. The magnitude on our specific AI metric is a new measurement.

Statistical test: two-sample t-test on close-bin normals vs. reference normals: t = 9.8, p < 1e-15, Cohen's d ≈ 1.5. Effect is real, not just statistically significant from a large N.

## **Key results**

Aggressive vehicles: mean AI = 71.4 (sd 8.3). Normals in heterogeneous run: mean AI = 33.1 (sd 6.7). Reference normals (aggressive-free): mean AI = 26.3. Population shockwave bump = +6.8 AI points.

Spatial decomposition: close normals 38.7 (+12.4 over reference), medium 32.9 (+6.6), far 27.8 (+1.5). Clean spatial decay — most of the population-level effect attributes to vehicles that came into proximity with an aggressive driver.

## **Plan for next week**

Two follow-ups requested by Daher. (1) Two-hop propagation: identify normal vehicles that came into proximity with normals that came into proximity with aggressives (two hops removed) and check whether they show elevated AI. (2) Aggressive-fraction sweep: 0%, 5%, 10%, 15%, 20% over 5-minute scenarios each, plot the dose-response curve. Also: make the final go/no-go call on CARLA.

VIPP 201A Weekly Reports  —  Hadi Al Shmaissani  —  Page  of
