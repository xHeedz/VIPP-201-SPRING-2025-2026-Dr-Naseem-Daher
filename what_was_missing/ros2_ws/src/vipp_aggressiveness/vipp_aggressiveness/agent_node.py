"""
AgentNode: /telemetry -> gate-head agent -> /env/aggressiveness_index.

RECREATED BY CLAUDE (Anthropic), September 2026. Not the original project code;
never built or run against ROS2 by its author. See what_was_missing/README.md.

Parameters
  model_path      TorchScript gate head (what_was_missing/model/gate_v2.pt)
  road_type       'highway' or 'urban'
  friction        road friction coefficient, 1.0 dry, 0.4 low friction
  road_length_km, lanes   used for the density term of the context vector
  latency_log     optional CSV path; ingest-to-publish latency per message
"""
import time

import rclpy
from rclpy.node import Node
from vipp_aggressiveness_msgs.msg import AggressivenessIndex, VehicleTelemetry

from vipp_aggressiveness.core import AgentRuntime, latency_stats


class AgentNode(Node):
    def __init__(self):
        super().__init__("agent_node")
        p = {name: self.declare_parameter(name, default).value for name, default in [
            ("model_path", ""), ("road_type", "highway"), ("friction", 1.0),
            ("road_length_km", 1.0), ("lanes", 3), ("latency_log", ""),
        ]}
        if not p["model_path"]:
            raise RuntimeError("set the model_path parameter to a TorchScript gate head")
        self.runtime = AgentRuntime(p["model_path"], p["road_type"], p["friction"],
                                    p["road_length_km"], p["lanes"])
        self.get_logger().info(f"loaded {p['model_path']} ({self.runtime.provenance.get('created_by', 'no provenance')})")
        self.pub = self.create_publisher(AggressivenessIndex, "/env/aggressiveness_index", 100)
        self.sub = self.create_subscription(VehicleTelemetry, "/telemetry", self.on_telemetry, 100)
        self.latency_log = p["latency_log"]
        self.latencies = []

    def on_telemetry(self, msg):
        t_in = time.perf_counter()
        now = self.get_clock().now().nanoseconds * 1e-9
        out = self.runtime.process(msg.vehicle_id, now, msg.speed, msg.accel, msg.prox, msg.wave, msg.slip)
        res = AggressivenessIndex()
        res.ai_score = float(out["ai_score"])
        res.ai_weather = float(out["ai_weather"])
        res.mixture_weights = [float(v) for v in out["mixture_weights"]]
        res.vehicle_id = msg.vehicle_id
        res.stamp = msg.stamp
        self.pub.publish(res)
        self.latencies.append((time.perf_counter() - t_in) * 1000.0)

    def report(self):
        stats = latency_stats(self.latencies)
        if stats:
            self.get_logger().info(f"in-node latency over {stats['n']} messages: median {stats['median_ms']:.2f} ms, "
                                   f"p99 {stats['p99_ms']:.2f} ms")
        if self.latency_log and self.latencies:
            with open(self.latency_log, "w") as f:
                f.write("latency_ms\n" + "\n".join(f"{v:.4f}" for v in self.latencies) + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
