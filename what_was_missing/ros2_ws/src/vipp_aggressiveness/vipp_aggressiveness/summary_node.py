"""
SummaryNode: /env/aggressiveness_index -> 1 Hz population summary on /env/aggressiveness_summary.

RECREATED BY CLAUDE (Anthropic), September 2026. Not the original project code;
never built or run against ROS2 by its author. See what_was_missing/README.md.

Keeps each vehicle's AI history for 2 s; vehicles not heard from in 2 s drop out.
"""
import rclpy
from rclpy.node import Node
from vipp_aggressiveness_msgs.msg import AggressivenessIndex, AggressivenessSummary

from vipp_aggressiveness.core import SummaryAggregator


class SummaryNode(Node):
    def __init__(self):
        super().__init__("summary_node")
        window = self.declare_parameter("window_s", 2.0).value
        self.agg = SummaryAggregator(window_s=window)
        self.sub = self.create_subscription(AggressivenessIndex, "/env/aggressiveness_index", self.on_index, 100)
        self.pub = self.create_publisher(AggressivenessSummary, "/env/aggressiveness_summary", 10)
        self.timer = self.create_timer(1.0, self.tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_index(self, msg):
        self.agg.add(msg.vehicle_id, self._now(), msg.ai_score)

    def tick(self):
        s = self.agg.tick(self._now())
        m = AggressivenessSummary()
        m.mean_ai, m.max_ai = float(s["mean_ai"]), float(s["max_ai"])
        m.n_vehicles, m.n_affected_neighbors = int(s["n_vehicles"]), int(s["n_affected_neighbors"])
        m.aggressive_fraction = float(s["aggressive_fraction"])
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SummaryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
