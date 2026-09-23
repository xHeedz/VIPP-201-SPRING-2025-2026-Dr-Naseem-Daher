"""
SumoEnv: a small highway-env-style wrapper around SUMO through TraCI.

RECREATED BY CLAUDE (Anthropic), September 2026, from the final report
(Section 2.4 and the repository description in Section 4.1). This is not the
original project code and was not used to produce any result in the reports.

API: reset() -> (obs, info), step(action) -> (obs, reward, terminated,
truncated, info), get_telemetry() -> {vehicle_id: dict}, close().

Scenarios (final report Section 2.4):
  highway       3-lane motorway, 1000 m, no on-ramps
  intersection  4-arm signalized intersection
  low_friction  the motorway with friction 0.4 on every edge

Driver types (final report values, plus the extra parameters marked below):
  aggressive    minGap 0.5 m, accel 4.0, impatience 1.0
  normal        SUMO defaults
  conservative  minGap 4.0 m, decel 3.5, impatience 0.0
The reports name IDM and MOBIL. SUMO has IDM but no MOBIL; its LC2013
lane-change model is used instead. speedFactor, tau and lane-change
eagerness differences between driver types are additions made here so the
three types actually drive differently.

SUMO has no tyre model, so edge friction alone changes nothing. In
low_friction every driver type also gets halved accel/decel and a longer
headway (additions made here). Slip is the kinematic proxy the reports
describe, |v_actual - v_intended| / v_actual, with v_intended the speed the
car-following model would choose given the current leader. Vehicles below
2 m/s or stopping for a red or yellow signal report zero slip, since that
deviation comes from the signal rather than the road surface. In testing the
proxy rose only slightly on the low-friction motorway, so it is a weak
friction signal and should be treated as one.

Ego actions:
  highway, low_friction: 0 LaneLeft, 1 Idle, 2 LaneRight, 3 Faster, 4 Slower
  intersection:          0 Brake, 1 Idle, 2 Accelerate
Reward: Social Latency (agent/social_latency.py) on the ego's AI.
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from itertools import count

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.aggressiveness import (  # noqa: E402
    REGIME_TARGETS, ai_from_lists, context_vector, normalize_features,
)
from agent.social_latency import SocialLatencyReward  # noqa: E402


def _sumo_home():
    home = os.environ.get("SUMO_HOME")
    if home and os.path.isdir(home):
        return home
    try:
        import sumo  # eclipse-sumo pip package
        return sumo.SUMO_HOME
    except ImportError:
        return None


_HOME = _sumo_home()
if _HOME and os.path.join(_HOME, "tools") not in sys.path:
    sys.path.append(os.path.join(_HOME, "tools"))
import traci  # noqa: E402


def _binary(name):
    exe = name + (".exe" if os.name == "nt" else "")
    if _HOME and os.path.exists(os.path.join(_HOME, "bin", exe)):
        return os.path.join(_HOME, "bin", exe)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found; install SUMO and set SUMO_HOME")


LANE_WIDTH = 3.2
_labels = count()

VTYPES = {
    "normal": 'carFollowModel="IDM" laneChangeModel="LC2013" color="0,0.6,1"',
    "aggressive": ('carFollowModel="IDM" laneChangeModel="LC2013" minGap="0.5" accel="4.0" '
                   'impatience="1.0" speedFactor="1.2" tau="0.6" lcAssertive="3" lcSpeedGain="3" color="1,0,0"'),
    "conservative": ('carFollowModel="IDM" laneChangeModel="LC2013" minGap="4.0" decel="3.5" '
                     'impatience="0.0" speedFactor="0.85" tau="1.8" lcSpeedGain="0.3" color="0,0.8,0"'),
    "ego": 'carFollowModel="IDM" laneChangeModel="LC2013" color="1,1,0"',
}
LOW_FRICTION_EXTRA = {  # accel, decel halved and longer headway
    "normal": 'accel="1.3" decel="2.25" tau="1.6"',
    "aggressive": 'accel="2.0" decel="2.25" tau="1.1"',   # replaces accel/tau above
    "conservative": 'accel="1.3" decel="1.75" tau="2.3"',
    "ego": 'accel="1.3" decel="2.25" tau="1.6"',
}


def _merge_attrs(base, extra):
    """Attribute string merge where extra overrides base."""
    def parse(s):
        out = {}
        for part in s.split('" '):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
        return out
    merged = parse(base)
    merged.update(parse(extra))
    return " ".join(f'{k}="{v}"' for k, v in merged.items())


class SumoEnv:
    SCENARIOS = ("highway", "intersection", "low_friction")

    def __init__(self, scenario="highway", gui=False, step_length=1.0 / 15.0,
                 flow_veh_per_hour=1800, aggressive_fraction=0.05, conservative_fraction=0.15,
                 warmup_s=40.0, max_steps=1500, seed=0, work_dir=None, weight_fn=None,
                 reward=None):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}")
        self.scenario = scenario
        self.gui = gui
        self.dt = step_length
        self.flow = flow_veh_per_hour
        self.frac_aggr = aggressive_fraction
        self.frac_cons = conservative_fraction
        self.warmup_s = warmup_s
        self.max_steps = max_steps
        self.seed = seed
        self.road_type = "urban" if scenario == "intersection" else "highway"
        self.friction = 0.4 if scenario == "low_friction" else 1.0
        regime = {"highway": "highway", "intersection": "urban", "low_friction": "weather"}[scenario]
        self.default_weights = REGIME_TARGETS[regime]
        # weight_fn(features, context) -> 4 weights; the DynamicWeightAgent can be plugged in here
        self.weight_fn = weight_fn or (lambda f, c: self.default_weights)
        self.reward_fn = reward or SocialLatencyReward()
        self.work_dir = work_dir or tempfile.mkdtemp(prefix=f"sumoenv_{scenario}_")
        os.makedirs(self.work_dir, exist_ok=True)
        self.cfg = self._build_scenario()
        self.conn = None
        self.label = f"sumoenv{next(_labels)}"
        self.ego = "ego"
        self._hist = {}
        self._target_speed = None
        self.steps = 0

    # ------------------------------------------------------------------ scenario files
    def _write(self, name, text):
        path = os.path.join(self.work_dir, name)
        with open(path, "w") as f:
            f.write(text)
        return path

    def _vtypes_xml(self):
        lines = []
        for vid, attrs in VTYPES.items():
            if self.scenario == "low_friction":
                attrs = _merge_attrs(attrs, LOW_FRICTION_EXTRA[vid])
            lines.append(f'    <vType id="{vid}" {attrs}/>')
        return "\n".join(lines)

    def _flows_xml(self, routes):
        """Split the total flow across routes and driver types."""
        shares = {"aggressive": self.frac_aggr, "conservative": self.frac_cons,
                  "normal": 1.0 - self.frac_aggr - self.frac_cons}
        out = []
        per_route = self.flow / len(routes)
        for rid, _ in routes:
            for vtype, share in shares.items():
                vph = per_route * share
                if vph <= 0:
                    continue
                out.append(f'    <flow id="f_{rid}_{vtype}" type="{vtype}" route="{rid}" begin="0" end="100000" '
                           f'vehsPerHour="{vph:.1f}" departLane="best" departSpeed="desired"/>')
        return "\n".join(out)

    def _build_scenario(self):
        netconvert = _binary("netconvert")
        if self.scenario in ("highway", "low_friction"):
            nod = self._write("net.nod.xml", '<nodes>\n  <node id="a" x="0" y="0"/>\n  <node id="b" x="1000" y="0"/>\n</nodes>\n')
            friction = f' friction="{self.friction}"' if self.scenario == "low_friction" else ""
            edg = self._write("net.edg.xml", f'<edges>\n  <edge id="hw" from="a" to="b" numLanes="3" speed="33.33"{friction}/>\n</edges>\n')
            routes = [("r_hw", "hw")]
            ego = '    <vehicle id="ego" type="ego" route="r_hw" depart="{d}" departLane="1" departPos="50" departSpeed="desired"/>'
        else:
            nod = self._write("net.nod.xml", (
                '<nodes>\n  <node id="c" x="0" y="0" type="traffic_light"/>\n'
                '  <node id="n" x="0" y="200"/>\n  <node id="s" x="0" y="-200"/>\n'
                '  <node id="e" x="200" y="0"/>\n  <node id="w" x="-200" y="0"/>\n</nodes>\n'))
            edges = []
            for arm in "nsew":
                edges.append(f'  <edge id="{arm}2c" from="{arm}" to="c" numLanes="2" speed="13.89"/>')
                edges.append(f'  <edge id="c2{arm}" from="c" to="{arm}" numLanes="2" speed="13.89"/>')
            edg = self._write("net.edg.xml", "<edges>\n" + "\n".join(edges) + "\n</edges>\n")
            routes = [(f"r_{a}{b}", f"{a}2c c2{b}") for a in "nsew" for b in "nsew" if a != b]
            ego = '    <vehicle id="ego" type="ego" route="r_we" depart="{d}" departLane="best" departSpeed="desired"/>'
        net = os.path.join(self.work_dir, "net.net.xml")
        cmd = [netconvert, "--node-files", nod, "--edge-files", edg, "-o", net, "--no-turnarounds", "true"]
        subprocess.run(cmd, check=True, capture_output=True)
        route_defs = "\n".join(f'    <route id="{rid}" edges="{edges}"/>' for rid, edges in routes)
        rou = self._write("net.rou.xml", (
            "<routes>\n" + self._vtypes_xml() + "\n" + route_defs + "\n" + self._flows_xml(routes) + "\n"
            + ego.format(d=f"{self.warmup_s:.1f}") + "\n</routes>\n"))
        cfg = self._write("scenario.sumocfg", (
            '<configuration>\n  <input>\n'
            f'    <net-file value="{os.path.basename(net)}"/>\n'
            f'    <route-files value="{os.path.basename(rou)}"/>\n'
            '  </input>\n  <time>\n'
            f'    <step-length value="{self.dt:.6f}"/>\n'
            '  </time>\n  <processing>\n'
            '    <collision.action value="warn"/>\n'
            '    <lanechange.duration value="2"/>\n'
            '  </processing>\n  <report>\n'
            '    <no-step-log value="true"/>\n    <no-warnings value="true"/>\n'
            '  </report>\n</configuration>\n'))
        return cfg

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed=None):
        self.close()
        binary = _binary("sumo-gui" if self.gui else "sumo")
        cmd = [binary, "-c", self.cfg, "--seed", str(self.seed if seed is None else seed)]
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
        self._hist = {}
        self._target_speed = None
        self.steps = 0
        self.reward_fn.reset()
        # run until the ego is inserted (traffic warms up first)
        for _ in range(int((self.warmup_s + 30.0) / self.dt)):
            self.conn.simulationStep()
            self._update_history()
            if self.ego in self.conn.vehicle.getIDList():
                break
        else:
            raise RuntimeError("ego vehicle was never inserted; lower the flow or raise warmup_s")
        self.conn.vehicle.setLaneChangeMode(self.ego, 512)   # only commanded changes, respecting gaps
        tel = self.get_telemetry()
        return self._obs(tel), {"telemetry": tel}

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    # ------------------------------------------------------------------ telemetry
    def _update_history(self):
        c = self.conn
        ids = set(c.vehicle.getIDList())
        for vid in list(self._hist):
            if vid not in ids:
                del self._hist[vid]
        window = max(2, int(round(1.0 / self.dt)))   # 1 s of history
        for vid in ids:
            edge = c.vehicle.getRoadID(vid)
            lat = c.vehicle.getLaneIndex(vid) * LANE_WIDTH + c.vehicle.getLateralLanePosition(vid)
            accel = c.vehicle.getAcceleration(vid)
            h = self._hist.get(vid)
            if h is None or h["edge"] != edge:
                h = {"edge": edge, "lat": deque(maxlen=window), "prev_accel": accel, "density": 0.0}
                self._hist[vid] = h
            h["jerk"] = (accel - h["prev_accel"]) / self.dt
            h["prev_accel"] = accel
            h["lat"].append(lat)

    def _density(self, vid):
        c = self.conn
        edge = c.vehicle.getRoadID(vid)
        h = self._hist[vid]
        if edge.startswith(":"):          # inside a junction: keep the last value
            return h["density"]
        lanes = c.edge.getLaneNumber(edge)
        length_km = c.lane.getLength(f"{edge}_0") / 1000.0
        h["density"] = c.edge.getLastStepVehicleNumber(edge) / max(length_km * lanes, 1e-6)
        return h["density"]

    def _vehicle(self, vid):
        c = self.conn
        speed = c.vehicle.getSpeed(vid)
        accel = c.vehicle.getAcceleration(vid)
        leader = c.vehicle.getLeader(vid, 100.0)
        has_leader = bool(leader) and bool(leader[0])
        gap = leader[1] if has_leader else 100.0
        allowed = c.vehicle.getAllowedSpeed(vid)
        v_int = allowed
        if has_leader:
            try:
                v_int = min(allowed, c.vehicle.getFollowSpeed(
                    vid, speed, gap, c.vehicle.getSpeed(leader[0]), c.vehicle.getDecel(leader[0]), leader[0]))
            except traci.TraCIException:
                pass
        # A vehicle stopping for a signal, or standing still, says nothing about grip
        stopping = any(d < 80.0 and st in "rRyY" for (_, _, d, st) in c.vehicle.getNextTLS(vid))
        slip = 0.0 if (stopping or speed < 2.0) else min(abs(speed - v_int) / speed, 1.0)
        h = self._hist[vid]
        wave = (max(h["lat"]) - min(h["lat"])) if h["lat"] else 0.0
        x, y = c.vehicle.getPosition(vid)
        density = self._density(vid)
        feats = normalize_features(speed, accel, gap, wave)
        ctx = context_vector(self.road_type, density, self.friction, slip)
        ai = ai_from_lists(feats, self.weight_fn(feats, ctx))
        return {"speed": speed, "accel": accel, "prox": gap, "wave": wave, "jerk": h["jerk"],
                "slip": slip, "x": x, "y": y, "type": c.vehicle.getTypeID(vid), "density": density,
                "features": feats, "context": ctx, "ai": ai}

    def get_telemetry(self):
        return {vid: self._vehicle(vid) for vid in self.conn.vehicle.getIDList()}

    def _obs(self, tel):
        e = tel.get(self.ego)
        return None if e is None else e["features"] + e["context"]

    # ------------------------------------------------------------------ control
    def _apply(self, action):
        c, ego = self.conn, self.ego
        v = c.vehicle.getSpeed(ego)
        if self.scenario == "intersection":
            if action == 0:
                c.vehicle.setSpeed(ego, max(v - 3.0, 0.0))
            elif action == 2:
                c.vehicle.setSpeed(ego, v + 2.0)
            else:
                c.vehicle.setSpeed(ego, -1)
            return
        if action == 0:
            c.vehicle.changeLaneRelative(ego, 1, 2.0)
        elif action == 2:
            c.vehicle.changeLaneRelative(ego, -1, 2.0)
        elif action == 3:
            self._target_speed = min(v + 2.0, c.vehicle.getAllowedSpeed(ego) * 1.2)
        elif action == 4:
            self._target_speed = max(v - 2.0, 0.0)
        if action in (3, 4):
            c.vehicle.setSpeed(ego, self._target_speed)
        elif action == 1:
            self._target_speed = None
            c.vehicle.setSpeed(ego, -1)

    def step(self, action=1):
        if self.ego in self.conn.vehicle.getIDList():
            self._apply(action)
        self.conn.simulationStep()
        self._update_history()
        self.steps += 1
        crashed = self.ego in self.conn.simulation.getCollidingVehiclesIDList()
        tel = self.get_telemetry()
        ego = tel.get(self.ego)
        reward, harm = 0.0, 0.0
        if ego is not None:
            others = [(t["x"], t["y"], t["ai"]) for vid, t in tel.items() if vid != self.ego]
            reward, harm = self.reward_fn(ego["ai"], (ego["x"], ego["y"]), others)
        terminated = crashed or ego is None
        truncated = self.steps >= self.max_steps
        info = {"telemetry": tel, "crashed": crashed, "harm": harm,
                "ai_ego": None if ego is None else ego["ai"]}
        return self._obs(tel), reward, terminated, truncated, info
