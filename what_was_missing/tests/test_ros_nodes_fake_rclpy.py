# Written by Claude (Anthropic), September 2026, as part of what_was_missing/. See ../README.md.
"""
Runs the three ROS2 nodes in-process against a minimal fake of rclpy and the
message package. This checks the node wiring (parameters, topics, callbacks,
timers) without ROS2 installed. It is not a substitute for a colcon build.
"""
import os
import sys
import types
from collections import defaultdict

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class _Clock:
    t = 0.0

    def now(self):
        c = self

        class _Now:
            nanoseconds = int(c.t * 1e9)

            def to_msg(self):
                return {"sec": int(c.t), "nanosec": int((c.t % 1) * 1e9)}
        return _Now()


CLOCK = _Clock()
BUS = defaultdict(list)          # topic -> callbacks
PUBLISHED = defaultdict(list)    # topic -> messages
TIMERS = []
PARAMS = {}


class _Param:
    def __init__(self, v):
        self.value = v


class _Pub:
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        PUBLISHED[self.topic].append(msg)
        for cb in BUS[self.topic]:
            cb(msg)


class _Timer:
    def __init__(self, period, cb):
        self.period, self.cb, self.active, self.next = period, cb, True, CLOCK.t + period

    def cancel(self):
        self.active = False


class _Logger:
    def info(self, *_):
        pass


class FakeNode:
    def __init__(self, name):
        self.name = name

    def declare_parameter(self, name, default):
        return _Param(PARAMS.get(name, default))

    def get_logger(self):
        return _Logger()

    def get_clock(self):
        return CLOCK

    def create_publisher(self, _type, topic, _qos):
        return _Pub(topic)

    def create_subscription(self, _type, topic, cb, _qos):
        BUS[topic].append(cb)

    def create_timer(self, period, cb):
        t = _Timer(period, cb)
        TIMERS.append(t)
        return t

    def destroy_node(self):
        pass


def _msg_class(name, fields):
    return type(name, (), {f: None for f in fields})


@pytest.fixture
def fake_ros(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = FakeNode
    rclpy.node = node_mod
    msgs = types.ModuleType("vipp_aggressiveness_msgs")
    msg = types.ModuleType("vipp_aggressiveness_msgs.msg")
    msg.VehicleTelemetry = _msg_class("VehicleTelemetry", ["speed", "accel", "prox", "wave", "jerk", "slip", "vehicle_id", "stamp"])
    msg.AggressivenessIndex = _msg_class("AggressivenessIndex", ["ai_score", "ai_weather", "mixture_weights", "vehicle_id", "stamp"])
    msg.AggressivenessSummary = _msg_class("AggressivenessSummary", ["mean_ai", "max_ai", "n_vehicles", "n_affected_neighbors", "aggressive_fraction"])
    msgs.msg = msg
    for name, mod in {"rclpy": rclpy, "rclpy.node": node_mod, "vipp_aggressiveness_msgs": msgs,
                      "vipp_aggressiveness_msgs.msg": msg}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for m in [k for k in sys.modules if k.startswith("vipp_aggressiveness.") and "node" in k]:
        monkeypatch.delitem(sys.modules, m)
    BUS.clear(), PUBLISHED.clear(), TIMERS.clear(), PARAMS.clear()
    CLOCK.t = 0.0
    yield


def test_pipeline_end_to_end(fake_ros):
    from vipp_aggressiveness.agent_node import AgentNode
    from vipp_aggressiveness.summary_node import SummaryNode
    from vipp_aggressiveness.telemetry_publisher_node import TelemetryPublisherNode

    PARAMS.update({"csv_path": os.path.join(ROOT, "results", "telemetry_highway.csv"),
                   "model_path": os.path.join(ROOT, "model", "gate_v2.pt")})
    agent, summary, pub = AgentNode(), SummaryNode(), TelemetryPublisherNode()
    # advance the fake clock for 5 simulated seconds, firing timers when due
    end = 5.0
    while CLOCK.t < end:
        due = min(t.next for t in TIMERS if t.active)
        CLOCK.t = due
        for t in TIMERS:
            if t.active and abs(t.next - due) < 1e-9:
                t.cb()
                t.next += t.period
    n_tel = len(PUBLISHED["/telemetry"])
    idx = PUBLISHED["/env/aggressiveness_index"]
    summ = PUBLISHED["/env/aggressiveness_summary"]
    assert n_tel > 0 and len(idx) == n_tel                     # one index message per telemetry message
    assert all(0.0 <= m.ai_score <= 100.0 and len(m.mixture_weights) == 3 for m in idx)
    assert len(summ) == 5 and summ[-1].n_vehicles > 0          # 1 Hz for 5 s
    assert idx[0].stamp == PUBLISHED["/telemetry"][0].stamp   # stamp passes through
    agent.report()
