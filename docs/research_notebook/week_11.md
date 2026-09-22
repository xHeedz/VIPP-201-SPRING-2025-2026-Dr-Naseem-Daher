# **Week 11: Tuesday Mar 31 – Monday Apr 6, 2026**

**Theme — ***Friction-aware regime. Slip-ratio telemetry. AI_Weather works.*

## **Goals for the week**

Implement the friction-aware regime that the midterm flagged. Add slip-ratio telemetry to the SUMO wrapper, set up a low-friction scenario (μ=0.4 instead of the default ~0.9, simulating wet/icy roads), verify the gate's weather mixture weight rises in response, tune the weather-target weights to be appropriate for low friction, and produce an AI_Weather formula that combines AI with a friction modifier.

## **Daily entries**

**Tue Mar 31***  —  Slip-ratio in SUMO.*

SUMO doesn't expose slip ratio directly — it uses friction as an edge attribute that affects the maximum-speed and following-distance behavior of vehicles, but doesn't compute kinematic slip per se. So I have to derive slip from kinematic observables. Used the formula slip ≈ |v_actual - v_intended| / v_actual, where v_intended is what the vehicle would have done absent friction limits (read from the IDM equilibrium speed for the current edge density) and v_actual is what TraCI reports.

On a high-friction edge (μ=0.9), slip is essentially zero (~0.01 noise floor). On a low-friction edge (μ=0.4), slip rises to 0.15-0.25 during accel events. Fed slip into the wrapper as an extra channel in the per-vehicle telemetry dict. About 25 lines of code. Works.

**Wed Apr 1***  —  Low-friction scenario + agent response.*

Built a low-friction SUMO scenario: same 3-lane motorway, but with μ=0.4 across all edges (no transition zones — uniform slippery road). Ran for 5 simulated minutes, ~600 vehicles. Mean slip ratio across the population: 0.12. Mean speed dropped from 96 km/h (high-friction reference) to 78 km/h, which is consistent with the IDM responding to lower friction with longer following distances and lower speeds.

Ran the gate-head agent on these telemetry samples in eval mode. Crucially, I added the slip ratio as a 5th context feature (was 4-dim before — now 5-dim). The gate's mixture-weight output: highway 0.42, urban 0.08, weather 0.50. The weather mixture weight rises dramatically in response to elevated slip — exactly the design intent. AI_Weather (the friction-modified score) for the population: mean 41.7, vs. 26.3 in high-friction reference. So the agent's gate, plus the closed-form AI weights, plus the new context feature, correctly diagnose the low-friction regime as more aggressive in the appropriate way.

**Thu Apr 2***  —  Weather-target tuning.*

The weather target weights I'd been using were hand-picked from the midterm's literature review (I'd written down (0.20, 0.30, 0.20, 0.30) — heavy on accel and wave). Tested whether those are actually the right targets for low-friction conditions by computing per-channel correlation with the slip-ratio observable, on the low-friction scenario.

Result: w_accel correlates strongly with slip (Pearson 0.71 — accel is a major slip driver), w_wave moderately (0.43 — wavy steering on slippery roads is dangerous), w_prox weakly (0.18 — proximity matters but less than acceleration on ice), w_speed essentially uncorrelated (0.04 — high speed is a problem on dry too). Adjusted the weather target to (0.10, 0.40, 0.15, 0.35), more emphasis on accel + wave, less on prox. Re-trained the gate-head agent for 200 epochs against the new target. Loss converges to 0.06. Saved as model/gate_v2.pt.

**Fri Apr 3***  —  AI_Weather composite formula.*

Wrote the AI_Weather composite formula. Two options: (a) modify the AI weights in low-friction conditions (which is what the gate already does via its mixture), or (b) add a friction multiplier on top of the standard AI score. Option (a) is what we have. Option (b) is what I prototyped today: AI_Weather = AI · (1 + γ · slip), with γ tuned so that AI_Weather on a fully-aggressive vehicle on a slippery road is 1.5× the AI on the same vehicle on a dry road. γ = 0.5 gives this approximately, but the value depends on context.

After thinking about it more: option (a) is cleaner and more interpretable — the four weights vary with regime, the AI formula itself stays the same. Option (b) is a hack on top. Going with (a) only. Documented the trade-off in a docstring on AggressivenessModel.

**Sat Apr 4***  —  Off.*

Off.

**Sun Apr 5***  —  Light test + wrap-up.*

Quick check: re-ran the heterogeneous-driver experiment from week 8 in the low-friction scenario (5% aggressive, μ=0.4). Population mean AI = 49.3. Reference normal in low-friction without aggressors: 41.7. So the shockwave bump in low friction is +7.6, which is slightly higher than the +6.8 we got in high friction. Aggressive drivers are *more* impactful on slippery roads because the kinematic perturbations they cause have higher consequences per unit input. Consistent intuition. Recorded as an interesting result for the report.

**Mon Apr 6***  —  Sync.*

Sync. Daher very happy with the friction-aware regime — said the slip-derivation trick is creative and practical. Likes that I went with option (a) instead of bolting on a friction multiplier. Two final asks: (a) make sure the slip computation is robust to edge cases (vehicle stopped, edge-end, vehicle just spawned) — there are some divide-by-zero conditions in my v_actual denominator that need guards. (b) Start the ROS2 wrapper for week 12. Final report drafting kicks off in earnest in week 13.

## **Reflection**

Friction-aware regime came together cleanly. The slip-ratio derivation was the only piece of real engineering — the rest was target tuning and verification. The fact that the gate naturally produces the right behavior (~50% weather mixture weight under elevated slip) without any modification to the architecture is a quiet but real validation of the gating design from week 4. Modular architecture pays off when you add a new regime and the existing pieces just work.

## **Plan for next week**

ROS2 wrapper. Build a minimal ROS2 node that subscribes to a /telemetry topic, runs the gate-head agent, publishes /env/aggressiveness_index. Use rclpy. Bag-record a 5-minute SUMO simulation pumping telemetry into the topic, verify AI score publishes at the rate we expect, replay with rosbag to confirm reproducibility.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
