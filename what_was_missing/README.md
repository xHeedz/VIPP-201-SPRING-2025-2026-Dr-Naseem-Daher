# what was missing

Everything in this folder was written by Claude (Anthropic) on 23 September 2026. It recreates components that the VIPP 201A final report, weekly reports and research notebook describe, but that exist nowhere in the project files. None of it is the original project code, and none of it produced any number, figure or table in `docs/`. The ROS2 packages have never been built or run against a real ROS2 installation.

Every Python file carries the same notice in its header, and both model checkpoints embed a provenance record (`created_by`, `created_on`, `not_the_original`, training data, final loss).

## contents

| component | described in | file | checked here |
|---|---|---|---|
| DynamicWeightAgent, MLP and gate heads | final report 2.1, 2.2, 3.1 | `agent/dynamic_weight_agent.py` | unit tests, training run |
| gating network checkpoint | weekly report 11 | `model/gate_v2.pt` | trained here, see findings |
| gating network, anchor only | not in the reports | `model/gate_anchor_only.pt` | trained here, see findings |
| Social Latency reward, fixed and Lagrangian | final report 3.2, addendum | `agent/social_latency.py` | unit tests, runs inside SumoEnv |
| SumoEnv, three scenarios | final report 2.4, 4.1 | `env/sumo_env.py` | runs all three scenarios in SUMO 1.27.1 |
| telemetry recorder | final report 2.6 (replay source) | `scripts/record_telemetry.py` | produced `results/telemetry_*.csv` |
| gate training | weekly reports 3 and 11 | `scripts/train_gate.py` | produced both checkpoints |
| ROS2 message types | weekly report 12, April monthly | `ros2_ws/src/vipp_aggressiveness_msgs/` | not built |
| AgentNode, TelemetryPublisherNode, SummaryNode, launch file | final report 2.6, weekly reports 12 to 14 | `ros2_ws/src/vipp_aggressiveness/` | fake-rclpy harness only |

The shared index definitions (regime targets, feature normalization, context vector) live in `agent/aggressiveness.py`, which has no PyTorch dependency so SumoEnv can use it on its own.

## tests

```
pip install -r requirements.txt
python -m pytest tests
```

18 tests, all passing at the time of writing. They cover the agent heads and loss, checkpoint export and provenance, the Social Latency clip and Lagrangian update, 30 steps of each SUMO scenario, agreement between the two copies of the index definitions, bit-identical scoring on a replayed 100-message slice, the summary window, and a 5-second run of all three ROS2 nodes wired together against a minimal fake of `rclpy`. That last test checks parameters, topics, callbacks and timers, but it is not a substitute for `colcon build` on a machine with ROS2.

## findings from the recreation

Training with the loss exactly as the final report states it (alpha annealed to 0.5 and held) does not keep the weights near the regime targets. On 60 s of SUMO telemetry per scenario, 200 epochs:

| regime | target | learned, alpha floor 0.5 | L1 to target | learned, alpha floor 1.0 |
|---|---|---|---|---|
| highway | (0.30, 0.10, 0.45, 0.15) | (0.18, 0.12, 0.53, 0.18) | 0.237 | matches, L1 below 0.001 |
| urban | (0.20, 0.15, 0.35, 0.30) | (0.21, 0.16, 0.28, 0.35) | 0.141 | matches, L1 below 0.001 |
| weather | (0.10, 0.40, 0.15, 0.35) | (0.06, 0.42, 0.16, 0.36) | 0.087 | matches, L1 below 0.001 |

With the floor at 0.5 the gate also stops separating regimes cleanly: on intersection telemetry its mean mixture is 0.44 highway, 0.37 urban, 0.20 weather. The cause is the RL term, `-E[1 - AI]`. It is roughly ten times larger than the KL term and is minimized by moving weight onto whichever features happen to be small, and the gate learns to route toward whichever regime yields the lowest score rather than the one that matches the context. `gate_v2.pt` follows the report's recipe; `gate_anchor_only.pt` is trained on the KL term alone and is the better choice for the ROS2 demo. Both training curves and gate handoff plots are in `results/`.

Per-message inference in `AgentRuntime` (context build, gate forward pass, both scores) measured 0.04 ms median and 0.08 ms p99 over 2,900 highway messages on one CPU core. This excludes ROS2 transport and serialization, which dominate end-to-end latency in practice.

The slip proxy is weak. Mean slip over 60 s was 0.007 on the dry motorway and 0.012 at friction 0.4, because SUMO has no tyre model and the proxy only sees car-following lag.

## choices made where the reports were ambiguous

- Input size: the final report's 9 inputs (4 kinematic, 5 context) rather than the 8 mentioned in week 2.
- Messages and nodes are two packages, since custom messages need an `ament_cmake` package and the nodes are `ament_python`. The reports describe a single package.
- SUMO has no MOBIL model, so LC2013 stands in. Speed factor, headway and lane-change eagerness differences between driver types, and the halved accel and decel on low friction, are additions.
- `wave` and `jerk` are computed upstream in SumoEnv and carried in `VehicleTelemetry`, so the node keeps no per-vehicle ring buffer.
- `dAI_neighbors` is the mean AI of vehicles within 50 m minus an exponential moving average of the same quantity (about 1 s time constant).
- The index is linear in each normalized feature, `100 * sum(w_i * n_i)`, since the agent's weights sum to one. `model/aggressiveness_model.py` at the repository root uses fixed weights and squared speed and proximity terms.
- `ai_weather` is the score under the weather regime's weights, whatever the gate predicts.
- `n_affected_neighbors` counts vehicles whose 2 s mean AI is more than one standard deviation above the population mean.

## not recreated

Measurements and recordings cannot be produced after the fact by writing code, so none are included: the 5-minute bag, the publish rate, the end-to-end latency figures, the bit-identical bag replay, the 50-vehicle load test, and the shockwave, dose-response and lambda sweep results. The code here makes each of them runnable.

## running

```
python scripts/record_telemetry.py --scenario highway --seconds 60        # also intersection, low_friction
python scripts/train_gate.py results/telemetry_*.csv                       # report recipe -> model/gate_v2.pt
python scripts/train_gate.py results/telemetry_*.csv --alpha-floor 1.0 --out model/gate_anchor_only.pt
```

`eclipse-sumo` ships the SUMO binaries; with a system install, set `SUMO_HOME` instead.

ROS2 (written for Humble, untested). PyTorch must be importable from the Python that ROS2 uses.

```
cd ros2_ws
colcon build
source install/setup.bash
ros2 launch vipp_aggressiveness pipeline.launch.py \
  csv_path:=$(realpath ../results/telemetry_highway.csv) \
  model_path:=$(realpath ../model/gate_anchor_only.pt) \
  latency_log:=/tmp/agent_latency.csv
ros2 topic hz /env/aggressiveness_index
ros2 topic echo /env/aggressiveness_summary
ros2 bag record /telemetry /env/aggressiveness_index
```
