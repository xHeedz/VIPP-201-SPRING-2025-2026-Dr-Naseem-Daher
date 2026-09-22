# **Week 7: Tuesday Mar 3 – Monday Mar 9, 2026**

**Theme — ***Long-horizon stability run + SUMO install in parallel.*

## **Goals for the week**

Run the stability check Daher asked for: 5000 episodes with curiosity on, predicted weights tracked over time, check that high-visitation regimes don't drift more than ~0.05 L1 from their target weights. In parallel: install SUMO on the lab machine, get the hello-world running, write a thin Python wrapper around TraCI that exposes the same step / get-vehicles / get-state surface as the highway-env wrapper so we can swap simulators by changing a config field.

## **Daily entries**

**Tue Mar 3***  —  Stability run kickoff + SUMO install.*

Launched the 5000-episode stability run on a tmux session. Should take ~6 hours. Logging predicted weights every 10 episodes into stability_run.csv.

While that ran, did the SUMO install. Followed the Eclipse SUMO docs: `sudo add-apt-repository ppa:sumo/stable`, `sudo apt-get update`, `sudo apt-get install sumo sumo-tools sumo-doc`. Version 1.20 installed clean. Verified the GUI works on a sample net: `sumo-gui -c tools/scenarios/sumo/2lane.sumocfg`. Cars drive past on a 2-lane road. Hello world.

**Wed Mar 4***  —  Stability results + TraCI exploration.*

Stability run finished overnight. Plotted predicted weights vs. episode for each regime. Highway weights drifted from (0.30, 0.10, 0.45, 0.15) at episode 1 to (0.31, 0.10, 0.45, 0.14) at episode 5000 — drift of 0.02 L1 over 4999 episodes, well inside the 0.05 budget. Urban and weather drifted by 0.04 and 0.03 L1 respectively. All three pass.

There's one interesting feature: highway shows a slow, monotonic 0.01-magnitude drift toward more weight on speed, which mirrors the embedding-drift experiment in week 4. Suggests the drift isn't noise — it's a small consistent signal that the agent finds slightly better weights than the hand-picked targets. I'm going to sit with that result rather than chase it further this term. Good motivator for the SUMO transition.

Started reading the TraCI Python tutorial. API is verbose but stable: traci.start(['sumo', '-c', 'cfg.sumocfg']), traci.simulationStep(), traci.vehicle.getIDList(), traci.vehicle.getSpeed(id), traci.vehicle.getPosition(id), and so on. Wrote a 30-line script that runs a 60-step sim and prints speeds — worked first try, which is rare enough to be worth recording.

**Thu Mar 5***  —  TraCI wrapper skeleton.*

Wrote env/sumo_wrapper.py. Same interface as the highway-env wrapper: reset(), step(action), close(), and a get_vehicles_telemetry() helper that returns a list of dicts (one per vehicle) with speed, accel, prox, wave, lane_id. The tricky part is proximity — in highway-env we relied on 1D position because lanes are parallel; in SUMO the road network can be arbitrary, so I have to project each vehicle onto its current edge and compute proximity along the edge.

Used the lane-internal coordinate that traci.vehicle.getLanePosition() returns, which is exactly what I needed: a scalar position along the current lane. Same-lane proximity is then just the difference in lane positions for vehicles on the same lane ID. Cross-lane proximity (for the urban regime later) is harder, but for the highway-equivalent scenario we're starting with, same-lane is enough.

**Fri Mar 6***  —  First end-to-end SUMO run.*

Built a 3-lane highway-equivalent scenario in SUMO: 5 km of straight 3-lane motorway, vehicles spawning from one end at a Poisson rate of 0.4/sec/lane, all set to a baseline speed of 100 km/h with sigma=0.1. Ran the wrapper for 60 simulated seconds, ~22 vehicles in the network at any given time, 8780 telemetry samples collected. Fed the samples through the AggressivenessModel and got AI scores in the same range as our highway-env runs (mean 18.4, 95th percentile 41.2).

Ran the gate-head DynamicWeightAgent against these samples in eval mode (no learning, just inference). Predicted weights cluster around (0.30, 0.11, 0.43, 0.16), within 0.02 L1 of the highway target. The gate puts ~85% of its mixture weight on highway, ~10% on urban, ~5% on weather. So the agent recognizes SUMO highway as a highway, which it should. First end-to-end SUMO run is a green build.

**Sat Mar 7***  —  Off (light reading).*

Off. Read parts of the SUMO 1.20 user manual on demand-modeling, specifically how to define heterogeneous vehicle types so I can later seed the simulation with a small fraction of "aggressive" drivers (lower minGap, higher accel, higher impatience) to replicate the shockwave experiment from the midterm's future-work section.

**Sun Mar 8***  —  Off.*

Off.

**Mon Mar 9***  —  Sync + scope for week 8.*

Sync. Daher's pleased that SUMO end-to-end works cleanly and the agent recognizes SUMO highway as highway without retraining. Wants the heterogeneous-driver experiment for week 8: seed the SUMO scenario with 5% aggressive vehicles defined via low minGap and high accel, run for several minutes of sim time, see whether the AI score field detects them. Bonus question: does the presence of aggressive drivers raise the average AI score of the *non-aggressive* drivers around them — the shockwave hypothesis.

## **Reflection**

Two parallel tracks both landed in one week. The stability run gave a clean answer (predicted weights drift very little over 5000 episodes), and the SUMO setup landed end-to-end on the first real attempt. TraCI has a steep documentation curve but a shallow practical curve once you know the four or five functions you actually need. The drift signal in the highway weights is still intriguing — too small to chase this term but worth a sentence in the final report.

## **Plan for next week**

Heterogeneous SUMO scenario: 5% aggressive vehicles (minGap=0.5, accel=4.0, decel=4.5, impatience=1.0), 95% normal vehicles. 5 minutes of sim time. Per-vehicle mean AI score. Confirm aggressive vehicles score higher; check whether neighboring normal vehicles score higher than reference normals in an aggressive-free run.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
