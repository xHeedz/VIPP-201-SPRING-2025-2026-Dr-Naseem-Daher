# VIPP-201 — Driver Aggressiveness Index (Spring 2025-2026, Dr. Naseem Daher)

Undergraduate research project modeling driver aggressiveness in context-aware, learning-driven autonomous ground vehicles for developing countries, using PyTorch, SUMO, and reinforcement learning.

## Structure

- `main.py` — entry point
- `src/agents/` — RL agent implementations (highway, urban, tuned, final versions)
- `src/analysis/` — aggressiveness index computation and result analysis
- `src/simulation/` — SUMO/Gym simulation environments and models
- `src/data_collection/` — telemetry and driving-behavior data collectors
- `src/visualization/` — network and brain/agent visualizations
- `src/reporting/` — presentation/validation report builders
- `sumo_configs/highway/` — SUMO network config for the highway scenario
- `sumo_configs/intersection/` — SUMO network config for the intersection scenario
- `data/` — CSV telemetry, training history, and driving-behavior datasets
- `data/logs/` — run logs
- `results/images/` — plots, charts, and screenshots
- `results/videos/` — demo/result recordings
- `docs/` — validation presentation
