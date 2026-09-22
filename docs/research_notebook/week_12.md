# **Week 12: Tuesday Apr 7 – Monday Apr 13, 2026**

**Theme — ***ROS2 wrapper. The end of the agent stack — now the system needs to talk to the rest of the world.*

## **Goals for the week**

Build a ROS2 node that wraps the agent + AI score: subscribes to a /telemetry topic, runs the gate-head agent on incoming messages, publishes the AI score on /env/aggressiveness_index. Use rclpy (Python ROS2 client). Bag-record a 5-minute SUMO sim pumping telemetry into the topic, replay to confirm reproducibility. Pass: messages publish at ≥10 Hz with median end-to-end latency < 50ms.

## **Daily entries**

**Tue Apr 7***  —  ROS2 setup + message types.*

ROS2 Humble was already installed on the lab machine from a previous project, which saved me probably half a day. Sourced /opt/ros/humble/setup.bash, ran `ros2 topic list` to verify it's working — empty list, good. Created a new colcon workspace at ~/ros2_ws and a package called `vipp_aggressiveness`.

Defined the message types. /telemetry uses a custom VehicleTelemetry.msg with fields: float32 speed, float32 accel, float32 prox, float32 wave, float32 jerk, float32 slip, string vehicle_id, builtin_interfaces/Time stamp. /env/aggressiveness_index uses a custom AggressivenessIndex.msg with fields: float32 ai_score, float32 ai_weather, float32[] mixture_weights (3 elements), string vehicle_id, builtin_interfaces/Time stamp. Built the package with colcon — first build failed because I had a stray comma in the CMakeLists, fixed in 30 seconds. Second build works.

**Wed Apr 8***  —  Subscriber + agent inference node.*

Wrote the AgentNode in agent_node.py. Subscribes to /telemetry, on each callback constructs the 5-dim context vector, runs the gate-head agent in inference mode (torch.no_grad(), .eval(), single-sample batch), computes AI score using AggressivenessModel, and publishes on /env/aggressiveness_index. The agent is loaded once at node construction time from model/gate_v2.pt.

Also wrote a TelemetryPublisherNode that reads a SUMO simulation log line by line and publishes telemetry messages at the rate they were recorded. Functions as a deterministic replay source so I can verify the agent node's output is reproducible.

First end-to-end test: launched both nodes in separate terminals, pumped 60 seconds of telemetry through. Ran `ros2 topic hz /env/aggressiveness_index` — getting 14.9 Hz, which matches the SUMO step rate (15 Hz) within sub-step jitter. End-to-end latency measured by stamping ingest time and publish time: median 18ms, 99th percentile 41ms. Both well under the 50ms budget. Pass.

**Thu Apr 9***  —  Bag recording + replay.*

Recorded a 5-minute bag with `ros2 bag record /telemetry /env/aggressiveness_index`. ~12 MB. Replayed with `ros2 bag play` and re-ran the AgentNode against the replayed telemetry. Output messages on /env/aggressiveness_index are bit-identical to the original recording (verified with a checksum on the AI score field over the full bag). Reproducibility confirmed.

Added a unit test that ingests a 100-message slice and checks that re-running the node produces identical output. Will catch any future non-determinism leak. Also good for demos in the final presentation: I can replay the bag without needing to keep SUMO running.

**Fri Apr 10***  —  Multi-vehicle handling.*

The current setup handles a single ego vehicle. The fleet experiments require many vehicles. Extended the AgentNode to handle multiple vehicle_ids: subscribe to a single /telemetry topic but route messages internally based on vehicle_id, maintain a small ring buffer of recent telemetry per vehicle (for the wave/jerk computation), publish per-vehicle AggressivenessIndex messages keyed by vehicle_id. Tested with 50 simulated vehicles — node still publishes at 15 Hz aggregated (so each vehicle gets ~0.3 Hz, which is fine for the 1Hz summary topic Daher wants in week 13).

Latency does grow with vehicle count: at 50 vehicles, 99th percentile end-to-end latency is 78ms. Above the 50ms budget, but only at high load. Profiled with cProfile — most of the time is in the agent forward pass (PyTorch overhead dominates at single-sample batches). Could batch the inference calls but that makes the per-vehicle latency hard to bound. Decision for now: at high vehicle counts, accept the latency growth and document. Can revisit by batching with a small temporal window if it becomes a problem.

**Sat Apr 11***  —  Off.*

Off.

**Sun Apr 12***  —  Documentation + README pass.*

Updated the README in the ros2_ws/src/vipp_aggressiveness package with: the message-type schemas, how to launch the agent node, how to launch the SUMO bridge, how to record a bag, how to replay. Wrote it as if I'd be the one reading it 6 months from now — more verbose than I'd usually write, with copy-pasteable shell commands. Future me will thank present me.

**Mon Apr 13***  —  Sync.*

Sync. Daher pleased with the ROS2 wrapper. Says the architecture (gate-head agent as the inference layer, ROS2 as the integration layer, SUMO as the simulator, AI score as the published signal) is the cleanest version of the project's story. Two requests for next week: (a) add a summary topic that aggregates per-vehicle AI scores into a population-level metric (mean AI, max AI, neighbor-affected count) at 1 Hz, so a downstream planner can subscribe to that lighter topic instead of all the per-vehicle ones. (b) Start the final report — Sections 1-3 (intro, related work, methods) are first, due as drafts by next Monday.

## **Reflection**

ROS2 felt like the easiest week of the term — partly because rclpy is mature and well-documented, partly because the agent stack underneath has been stable for weeks. The hardest part was actually the bag-replay reproducibility check, which surfaced the non-determinism risks I want to head off before they become bugs. Latency budgets are met for single-vehicle and modest fleet sizes; the high-fleet latency growth is documented but not addressed.

## **Plan for next week**

Add the summary topic at 1Hz. Begin final report drafting — Sections 1-3 (intro, related work / motivation, methods including the closed-form AI formulation, gate architecture, curiosity bonus, and Social Latency penalty).

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
