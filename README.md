# driver aggressiveness index

VIPP 201A undergraduate research project, American University of Beirut, 2025 to 2026.
Supervisor: Dr. Naseem Daher. Author: Hadi Al Shmaissani.

A closed-form Aggressiveness Index (AI) scores a driver from four kinematic signals: speed, acceleration, proximity to the lead vehicle, and lateral waviness. The index is used to label NPC traffic in SUMO highway and intersection scenarios and as a reward signal for reinforcement learning agents trained in highway-env and SUMO.

## the index

Each signal is normalized to [0, 1]:

- speed: v / 150 km/h
- acceleration: |a| / 5 m/s^2
- proximity: 1 - d / 50 m when the lead vehicle is within 50 m, otherwise 0
- waviness: |lateral drift| / 1.5 m

```
AI = min(100, 100 * (w_s * n_speed^2 + w_a * n_accel + w_p * n_prox^2 + w_w * n_wave))
```

Scores below 35 are labeled conservative, 35 up to 70 normal, and 70 or above aggressive. The implementation lives in `model/aggressiveness_model.py`.

## validation

Accuracy is measured by scoring every vehicle twice. `GroundTruthAssessor` reads simulation internals directly. `AgentAssessor` sees only the observation vector with sensor noise added, and never reads the ground truth. The two paths stay separate so agreement between them means something, and category agreement is reported per scenario at the end of each run. Both live in `model/aggressiveness_core.py`, with the SUMO equivalents in `scripts/sumo_runner.py`.

## repository layout

```
model/                  closed-form AI (AggressivenessModel) and the two-track assessors
agent/                  tabular Q-learner (DriverQLearner) and the shared reward function
env/scenarios/          highway, urban and weather RL scenarios in highway-env
env/sumo_scenarios/     SUMO networks and routes for the highway and intersection scenarios
scripts/                entry points: SUMO agents, validation runner, data collection, plots
scripts/highway_env/    PyTorch policy agents trained in highway-env
experiments/            dynamic weight agent experiments on synthetic telemetry
legacy/                 phase 1 scripts kept for reference
data/                   telemetry, NPC aggressiveness datasets, training histories, run logs
results/figures/        plots and charts
results/videos/         simulation and dashboard recordings
docs/                   final and progress reports, weekly and monthly reports, research notebook, bibliography, slides
paths.py                shared folder locations used by every script
what_was_missing/       components the reports describe but the project files lack, recreated by Claude in September 2026
```

All scripts resolve paths through `paths.py`, so they can be launched from any working directory.

`what_was_missing/` is self-contained, with its own tests and requirements. Nothing in it is original project code, and nothing in it produced any result in `docs/`; its README lists what each file recreates, what was verified, and what could not be recreated.

## setup

Python 3.11.

```
pip install -r requirements.txt
```

The SUMO scripts need SUMO installed (tested with 1.26) and the `SUMO_HOME` environment variable pointing to the installation folder, since TraCI is loaded from `$SUMO_HOME/tools`.

## running

SUMO

```
python scripts/build_networks.py                 # compile .nod/.edg into .net.xml (needs netconvert)
python scripts/sumo_highway_agent.py --no-gui    # 200-epoch highway agent, logs NPC aggressiveness
python scripts/sumo_urban_agent.py --no-gui      # 200-epoch intersection agent
python scripts/collect_data.py                   # 5-epoch headless collection with full hand-calculation log
python scripts/build_validation_ppt.py           # rebuild docs/slides/aggressiveness_validation.pptx
python scripts/sumo_runner.py                    # ground truth vs agent assessor, all three scenarios
```

`sumo_runner.py` generates its own networks, routes and configs under `env/sumo_scenarios/validation/`, so it needs netconvert but not `build_networks.py`.

Omitting `--no-gui` opens sumo-gui.

highway-env

```
python scripts/simulator.py                      # demo run, writes data/demo_data.csv
python agent/driver_q_learner.py                 # Q-learner on data/demo_data.csv
python env/scenarios/highway_scenario.py         # policy training with noisy assessment and accuracy report
python env/scenarios/urban_scenario.py
python env/scenarios/weather_scenario.py         # wet or icy roads, elevated sensor noise
python scripts/highway_env/rl_agent_baseline.py
python scripts/highway_env/rl_agent_tuned.py     # exploration decay, entropy bonus, tanh state scaling
python scripts/highway_env/urban_agent.py
```

plots

```
python scripts/plot_demo_results.py
python scripts/plot_training_dashboard.py
python scripts/brain_visualizer.py               # interactive network visualization
```

## documents

reports

- [final report, spring 2026](docs/reports/final_report_spring2026.pdf)
- [addendum on hyperparameter justification](docs/reports/addendum_hyperparameter_fixes.md) ([pdf](docs/reports/addendum_hyperparameter_fixes.pdf))
- [speaker script for the final presentation](docs/reports/speaker_script_spring2026.md) ([pdf](docs/reports/speaker_script_spring2026.pdf))
- [milestone 1 progress report](docs/reports/milestone1_progress_report.pdf)
- [final 4-day report, Apr 28 to May 1](docs/reports/final_4_day_report_apr28_may01.md)
- [weekly reports](docs/reports/weekly) and [monthly reports](docs/reports/monthly)
- [research notebook](docs/research_notebook)
- [annotated bibliography](docs/bibliography)
- [submission index for the spring term](docs/submission_index.md)

slides

- [final presentation, spring 2026](docs/slides/final_presentation_spring2026.pdf) ([pptx](docs/slides/final_presentation_spring2026.pptx))
- [project recap, september 2026](docs/slides/project_recap_september2026.pdf) ([keynote](docs/slides/project_recap_september2026.key))
- [milestone 1 presentation](docs/slides/milestone1_presentation.pdf)
- [aggressiveness validation deck](docs/slides/aggressiveness_validation.pptx), generated by `scripts/build_validation_ppt.py`

The PDF copies of the presentation and the recap are LibreOffice exports of the source files next to them.
