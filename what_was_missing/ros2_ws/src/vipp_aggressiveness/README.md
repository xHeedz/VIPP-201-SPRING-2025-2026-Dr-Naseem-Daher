# vipp_aggressiveness

ROS2 nodes recreated by Claude (Anthropic) in September 2026. Not the original project code, and never built against ROS2 by its author. Build, launch and bag commands are in `what_was_missing/README.md`.

Topics

- `/telemetry` (`VehicleTelemetry`): raw per-vehicle kinematics, from `telemetry_publisher_node` replaying a CSV
- `/env/aggressiveness_index` (`AggressivenessIndex`): per-vehicle score from `agent_node`
- `/env/aggressiveness_summary` (`AggressivenessSummary`): 1 Hz population aggregates from `summary_node`
