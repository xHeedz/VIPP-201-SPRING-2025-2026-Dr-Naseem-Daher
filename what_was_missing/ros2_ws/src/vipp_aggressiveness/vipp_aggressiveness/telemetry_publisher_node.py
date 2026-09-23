"""
TelemetryPublisherNode: replays a SUMO telemetry CSV on /telemetry at the recorded rate.

RECREATED BY CLAUDE (Anthropic), September 2026. Not the original project code;
never built or run against ROS2 by its author. See what_was_missing/README.md.

The CSV comes from what_was_missing/scripts/record_telemetry.py.

Parameters
  csv_path   telemetry CSV
  loop       restart from the top when the file ends
"""
import rclpy
from rclpy.node import Node
from vipp_aggressiveness_msgs.msg import VehicleTelemetry

from vipp_aggressiveness.core import load_telemetry_csv


class TelemetryPublisherNode(Node):
    def __init__(self):
        super().__init__("telemetry_publisher_node")
        path = self.declare_parameter("csv_path", "").value
        self.loop = self.declare_parameter("loop", False).value
        if not path:
            raise RuntimeError("set the csv_path parameter")
        self.steps = load_telemetry_csv(path)
        if len(self.steps) < 2:
            raise RuntimeError("telemetry CSV needs at least two time steps")
        dt = self.steps[1][0] - self.steps[0][0]
        self.pub = self.create_publisher(VehicleTelemetry, "/telemetry", 100)
        self.i = 0
        self.timer = self.create_timer(dt, self.tick)
        self.get_logger().info(f"replaying {len(self.steps)} steps at {1.0 / dt:.1f} Hz from {path}")

    def tick(self):
        if self.i >= len(self.steps):
            if not self.loop:
                self.get_logger().info("replay finished")
                self.timer.cancel()
                return
            self.i = 0
        stamp = self.get_clock().now().to_msg()
        for r in self.steps[self.i][1]:
            m = VehicleTelemetry()
            m.speed, m.accel, m.prox = float(r["speed"]), float(r["accel"]), float(r["prox"])
            m.wave, m.jerk, m.slip = float(r["wave"]), float(r["jerk"]), float(r["slip"])
            m.vehicle_id = r["vehicle_id"]
            m.stamp = stamp
            self.pub.publish(m)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryPublisherNode()
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
