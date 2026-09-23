# Written by Claude (Anthropic), September 2026, as part of what_was_missing/. See ../README.md.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROS_PKG = os.path.join(ROOT, "ros2_ws", "src", "vipp_aggressiveness")
for p in (ROOT, ROS_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)
