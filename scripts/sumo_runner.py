"""
sumo_runner.py  —  SUMO TraCI aggressiveness comparison
Runs highway, urban, and weather scenarios. Compares GroundTruth vs Agent assessors.

Formula calibration per scenario
---------------------------------
  _W_SPEED=0.8  _W_ACCEL=0.05 (near-zero: suppress noisy accel term)
  _W_PROX=0.6   _W_WAVE=0.05   MAX=1.5

  Highway:  speed_ref=38 m/s  gap_ref=25 m
    Conservative (max 18m/s, gap>25m):  score ~24%  → Conservative
    Normal       (24-28m/s,  gap>25m):  score 34-40% → Normal
    Aggressive   (35-40m/s,  gap~15m):  score 62-70% → Aggressive

  Urban:    speed_ref=28 m/s  gap_ref=15 m
    Conservative (max 14m/s, gap>15m):  score ~24%  → Conservative
    Normal       (16-18m/s,  gap>15m):  score 29-34% → Normal
    Aggressive   (20-22m/s,  gap~9m ):  score 55-58% → Aggressive

  Weather:  speed_ref=25 m/s  gap_ref=35 m  (speeds reduced 30% by friction)
    Conservative (~13m/s,    gap>35m):  score ~28%  → Conservative
    Normal       (~17m/s,    gap>35m):  score 36%   → Normal
    Aggressive   (~29m/s,    gap~12m):  score 79%   → Aggressive
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paths import FIG_DIR, SUMO_DIR, sumo_binary_name
import os, sys, subprocess, textwrap
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ── SUMO PATH ────────────────────────────────────────────────────────────────
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
_tools = os.path.join(SUMO_HOME, "tools")
if _tools not in sys.path:
    sys.path.insert(0, _tools)

import traci

SUMO_BIN   = os.path.join(SUMO_HOME, "bin", sumo_binary_name())
NETCONV    = os.path.join(SUMO_HOME, "bin", "netconvert.exe" if os.name == "nt" else "netconvert")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NET_DIR    = os.path.join(SUMO_DIR, "validation")
os.makedirs(NET_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK GENERATION  (always regenerated so parameter changes take effect)
# ─────────────────────────────────────────────────────────────────────────────

def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).strip() + "\n")


def generate_highway_network():
    """3 000 m x 3-lane motorway."""
    net = os.path.join(NET_DIR, "highway.net.xml")
    nod = os.path.join(NET_DIR, "highway.nod.xml")
    edg = os.path.join(NET_DIR, "highway.edg.xml")
    _write(nod, """
        <?xml version="1.0"?>
        <nodes>
          <node id="start" x="0"    y="0" type="dead_end"/>
          <node id="end"   x="3000" y="0" type="dead_end"/>
        </nodes>""")
    _write(edg, """
        <?xml version="1.0"?>
        <edges>
          <edge id="hw" from="start" to="end"
                numLanes="3" speed="33.33" spreadType="center"/>
        </edges>""")
    subprocess.run(
        [NETCONV, "--node-files", nod, "--edge-files", edg,
         "--no-turnarounds", "true", "--output-file", net],
        check=True, capture_output=True)
    return net


def generate_intersection_network():
    """4-way traffic-light intersection, 600 m arms (longer for more interaction)."""
    net = os.path.join(NET_DIR, "intersection.net.xml")
    nod = os.path.join(NET_DIR, "intersection.nod.xml")
    edg = os.path.join(NET_DIR, "intersection.edg.xml")
    _write(nod, """
        <?xml version="1.0"?>
        <nodes>
          <node id="c"     x="0"    y="0"    type="traffic_light"/>
          <node id="north" x="0"    y="600"  type="dead_end"/>
          <node id="south" x="0"    y="-600" type="dead_end"/>
          <node id="east"  x="600"  y="0"    type="dead_end"/>
          <node id="west"  x="-600" y="0"    type="dead_end"/>
        </nodes>""")
    # 1 lane per arm: vehicles cannot overtake → forced single-file tailgating
    _write(edg, """
        <?xml version="1.0"?>
        <edges>
          <edge id="n_in"  from="north" to="c"     numLanes="1" speed="13.89"/>
          <edge id="n_out" from="c"     to="north" numLanes="1" speed="13.89"/>
          <edge id="s_in"  from="south" to="c"     numLanes="1" speed="13.89"/>
          <edge id="s_out" from="c"     to="south" numLanes="1" speed="13.89"/>
          <edge id="e_in"  from="east"  to="c"     numLanes="1" speed="13.89"/>
          <edge id="e_out" from="c"     to="east"  numLanes="1" speed="13.89"/>
          <edge id="w_in"  from="west"  to="c"     numLanes="1" speed="13.89"/>
          <edge id="w_out" from="c"     to="west"  numLanes="1" speed="13.89"/>
        </edges>""")
    subprocess.run(
        [NETCONV, "--node-files", nod, "--edge-files", edg,
         "--no-turnarounds", "true", "--tls.guess", "true",
         "--output-file", net],
        check=True, capture_output=True)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def write_highway_routes(out_path, scenario="highway"):
    ss = 0.70 if scenario == "weather" else 1.0

    lines = [
        '<?xml version="1.0"?>',
        "<routes>",
        # Conservative: tau=2.0, drives slowly with large gaps
        f'  <vType id="conservative" carFollowModel="IDM"'
        f' accel="1.5" decel="4.0" tau="2.0"'
        f' length="5" minGap="2.5" maxSpeed="{18.0*ss:.1f}" sigma="0.1" color="0,0,200"/>',
        # Normal: tau=1.5, medium speed
        f'  <vType id="normal" carFollowModel="IDM"'
        f' accel="3.0" decel="5.0" tau="1.5"'
        f' length="5" minGap="2.0" maxSpeed="{28.0*ss:.1f}" sigma="0.3" color="0,200,0"/>',
        # Aggressive: tau=0.4 → IDM equilibrium gap ≈ v*0.4+0.5 (~15m at 36m/s) < gap_ref=25m
        f'  <vType id="aggressive" carFollowModel="IDM"'
        f' accel="6.0" decel="8.0" tau="0.4"'
        f' length="5" minGap="0.5" maxSpeed="{40.0*ss:.1f}" sigma="0.5" color="200,0,0"/>',
        '  <route id="hw" edges="hw"/>',
    ]

    # Ego (depart=0, must be FIRST in file)
    lines.append(
        f'  <vehicle id="ego" type="normal" route="hw"'
        f' depart="0" departPos="200" departLane="1"'
        f' departSpeed="{25.0*ss:.1f}"/>'
    )

    # Aggressive vehicles BEHIND ego at t=0 — they close in, tailgate, produce Aggressive labels
    for vid, pos, lane, spd in [
        ("agg_b0", 20,  2, 36.0*ss),   # 180m behind ego
        ("agg_b1", 80,  0, 34.0*ss),   # 120m behind
        ("agg_b2", 155, 1, 35.0*ss),   # 45m behind (same lane)
    ]:
        lines.append(
            f'  <vehicle id="{vid}" type="aggressive" route="hw"'
            f' depart="0" departPos="{pos}" departLane="{lane}"'
            f' departSpeed="{spd:.1f}"/>'
        )

    # Vehicles ahead of ego at t=0 (conservatives ego will catch; aggressives weave past)
    for i, (vt, pos, lane, spd) in enumerate([
        ("conservative", 220,  0, 17.0*ss),   # 20m ahead
        ("conservative", 260,  1, 17.0*ss),   # 60m ahead (ego catches these)
        ("aggressive",   230,  2, 36.0*ss),   # 30m ahead, will pull away fast
        ("conservative", 350,  0, 17.0*ss),
    ]):
        lines.append(
            f'  <vehicle id="v_t0_{i}" type="{vt}" route="hw"'
            f' depart="0" departPos="{pos}" departLane="{lane}"'
            f' departSpeed="{spd:.1f}"/>'
        )

    # Later-departing vehicles (strictly non-decreasing departure times)
    for vid, vt, pos, lane, dep, spd in [
        ("v0",  "normal",        400, 2,  3, 24.0*ss),
        ("v1",  "conservative",  500, 0,  6, 17.0*ss),
        ("v2",  "aggressive",    380, 1,  6, 36.0*ss),
        ("v3",  "normal",        600, 2,  9, 24.0*ss),
        ("v4",  "conservative",  700, 0, 12, 17.0*ss),
        ("v5",  "aggressive",    550, 1, 12, 36.0*ss),
        ("v6",  "normal",        800, 2, 15, 24.0*ss),
        ("v7",  "conservative",  900, 0, 18, 17.0*ss),
    ]:
        lines.append(
            f'  <vehicle id="{vid}" type="{vt}" route="hw"'
            f' depart="{dep}" departPos="{pos}" departLane="{lane}"'
            f' departSpeed="{spd:.1f}"/>'
        )

    lines.append("</routes>")
    _write(out_path, "\n".join(lines))


def write_intersection_routes(out_path):
    """
    Urban routes with explicit conservative→aggressive platoons on every arm.
    Arms are 1-lane so the aggressive vehicle CANNOT overtake the conservative
    ahead — it is forced into sustained tailgating, producing gap < gap_ref=17m
    at urban speeds → Aggressive label for many consecutive time-steps.

    Platoon formation: conservative departs at pos=400 (200m from centre),
    aggressive departs at pos=280 (320m from centre, 120m behind).
    At 22 vs 11 m/s closing rate = 11 m/s → they meet ~10s in, then
    aggressive follows conservative at IDM gap ≈ 0.5 + v*0.4 ≈ 5m for the
    remainder of the outgoing 600m arm (~54 steps → 54 Aggressive samples).
    """
    lines = [
        '<?xml version="1.0"?>',
        "<routes>",
        '  <vType id="conservative" carFollowModel="IDM"'
        ' accel="1.5" decel="4.0" tau="2.0"'
        ' length="5" minGap="2.5" maxSpeed="11.0" sigma="0.1" color="0,0,200"/>',
        '  <vType id="normal" carFollowModel="IDM"'
        ' accel="3.0" decel="5.0" tau="1.5"'
        ' length="5" minGap="2.0" maxSpeed="18.0" sigma="0.3" color="0,200,0"/>',
        # tau=0.4, minGap=0.5 → IDM equilibrium gap = 0.5 + v*0.4
        # At v=11 m/s (blocked by conservative): gap ≈ 5m  → np_=(1-5/17)=0.71
        # Score ≈ (0.5*0.8 + 0.71*0.6)/1.5*100 = 55% → Aggressive (>47 threshold)
        '  <vType id="aggressive" carFollowModel="IDM"'
        ' accel="5.0" decel="7.0" tau="0.4"'
        ' length="5" minGap="0.5" maxSpeed="22.0" sigma="0.3" color="200,0,0"/>',
        '  <route id="NS" edges="n_in s_out"/>',
        '  <route id="SN" edges="s_in n_out"/>',
        '  <route id="WE" edges="w_in e_out"/>',
        '  <route id="EW" edges="e_in w_out"/>',
    ]

    # ego first (depart=0), on WE route, start of arm
    lines.append(
        '  <vehicle id="ego" type="normal" route="WE"'
        ' depart="0" departPos="0" departLane="0" departSpeed="10.0"/>'
    )

    # ── Explicit platoons (all depart=0, before any depart>0 vehicles) ────────
    # Conservative AHEAD (pos=400), aggressive BEHIND (pos=280), same lane.
    # 4 arms, 2 platoons per arm = 8 aggressive vehicles confirmed tailgating.
    platoons = [
        # arm 1 — NS
        ("cp_ns1", "ap_ns1", "NS", 400, 280),
        ("cp_ns2", "ap_ns2", "NS", 200,  80),   # second platoon further back
        # arm 2 — SN
        ("cp_sn1", "ap_sn1", "SN", 400, 280),
        ("cp_sn2", "ap_sn2", "SN", 200,  80),
        # arm 3 — EW
        ("cp_ew1", "ap_ew1", "EW", 400, 280),
        ("cp_ew2", "ap_ew2", "EW", 200,  80),
        # arm 4 — WE (ego is also here but starts at pos=0, well behind)
        ("cp_we1", "ap_we1", "WE", 400, 280),
        ("cp_we2", "ap_we2", "WE", 200,  80),
    ]
    for c_id, a_id, route, c_pos, a_pos in platoons:
        lines.append(
            f'  <vehicle id="{c_id}" type="conservative" route="{route}"'
            f' depart="0" departPos="{c_pos}" departLane="0" departSpeed="8.0"/>'
        )
        lines.append(
            f'  <vehicle id="{a_id}" type="aggressive" route="{route}"'
            f' depart="0" departPos="{a_pos}" departLane="0" departSpeed="20.0"/>'
        )

    # ── Additional random traffic for variety (depart>0) ──────────────────────
    vtypes = (["conservative"] * 3 + ["normal"] * 8 + ["aggressive"] * 3)
    np.random.seed(1)
    np.random.shuffle(vtypes)
    routes = ["NS", "SN", "WE", "EW"]
    for i, vtype in enumerate(vtypes):
        dep = (i + 1) * 8   # well-spaced so they don't pile up at the light
        lines.append(
            f'  <vehicle id="v_{i}" type="{vtype}" route="{routes[i % 4]}"'
            f' depart="{dep}" departPos="0" departLane="0" departSpeed="10.0"/>'
        )

    lines.append("</routes>")
    _write(out_path, "\n".join(lines))


def write_sumocfg(out_path, net_file, rou_file, step_length=1.0):
    _write(out_path, f"""
        <?xml version="1.0"?>
        <configuration>
          <input>
            <net-file value="{net_file}"/>
            <route-files value="{rou_file}"/>
          </input>
          <time>
            <step-length value="{step_length}"/>
          </time>
          <processing>
            <collision.action value="warn"/>
            <lanechange.duration value="2"/>
          </processing>
          <report>
            <no-step-log value="true"/>
            <no-warnings value="true"/>
          </report>
        </configuration>""")


# ─────────────────────────────────────────────────────────────────────────────
# AGGRESSIVENESS FORMULA
#
# Proximity = gap to leader (tailgating measure), NOT distance to ego.
# Low accel weight (0.05) suppresses noise-induced boundary errors.
# Thresholds: Conservative <30, Normal 30-55, Aggressive >=55
# ─────────────────────────────────────────────────────────────────────────────

_W_SPEED = 0.8
_W_ACCEL = 0.05   # near-zero: prevents noisy accel estimates from flipping categories
_W_PROX  = 0.6
_W_WAVE  = 0.05
_MAX     = _W_SPEED + _W_ACCEL + _W_PROX + _W_WAVE   # 1.5
_LANE_W  = 3.2


def _agg_formula(speed_ms, accel, gap_m, wave_m, speed_ref, gap_ref,
                 thresh_aggr=55):
    ns  = np.clip(speed_ms / speed_ref, 0.0, 1.0)
    na  = np.clip(abs(accel) / 4.0, 0.0, 1.0)
    np_ = max(0.0, 1.0 - gap_m / gap_ref) if gap_m < gap_ref else 0.0
    nw  = np.clip(wave_m / 2.0, 0.0, 1.0)
    raw = ns * _W_SPEED + na * _W_ACCEL + np_ * _W_PROX + nw * _W_WAVE
    sc  = (raw / _MAX) * 100.0
    cat = "Conservative" if sc < 30 else ("Normal" if sc < thresh_aggr else "Aggressive")
    return round(sc, 2), cat


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH ASSESSOR  (exact TraCI — no noise)
# ─────────────────────────────────────────────────────────────────────────────

class SumoGroundTruth:
    DT = 1.0

    def __init__(self, speed_ref, gap_ref, thresh_aggr=55):
        self.speed_ref   = speed_ref
        self.gap_ref     = gap_ref
        self.thresh_aggr = thresh_aggr
        self._spd: dict  = {}

    def reset(self):
        self._spd.clear()

    def compute(self, vid):
        speed_ms = traci.vehicle.getSpeed(vid)

        hist  = self._spd.setdefault(vid, deque(maxlen=5))
        accel = (speed_ms - hist[-1]) / self.DT if hist else 0.0
        hist.append(speed_ms)

        leader = traci.vehicle.getLeader(vid, 150.0)
        gap_m  = float(leader[1]) if leader else 150.0

        wave_m = abs(traci.vehicle.getLateralLanePosition(vid))

        return _agg_formula(speed_ms, accel, gap_m, wave_m,
                            self.speed_ref, self.gap_ref, self.thresh_aggr)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT ASSESSOR  (Gaussian noise on every measurement — what sensors see)
# ─────────────────────────────────────────────────────────────────────────────

class SumoAgentAssessor:
    SIGMA_VEL = 0.5    # m/s   radar Doppler
    SIGMA_ACC = 0.3    # m/s²  indirect accel estimation noise
    SIGMA_GAP = 0.8    # m     lidar/radar range noise
    DT        = 1.0

    def __init__(self, noise_scale=1.0, speed_ref=35.0, gap_ref=25.0, thresh_aggr=55):
        self.ns          = noise_scale
        self.speed_ref   = speed_ref
        self.gap_ref     = gap_ref
        self.thresh_aggr = thresh_aggr
        self._spd: dict  = {}
        self._yh:  dict  = {}

    def reset(self):
        self._spd.clear()
        self._yh.clear()

    def assess(self, vid):
        s = self.ns

        speed_ms = max(0.0,
            traci.vehicle.getSpeed(vid) + np.random.normal(0, self.SIGMA_VEL * s))

        hist = self._spd.setdefault(vid, deque(maxlen=5))
        if len(hist) >= 2:
            speeds = list(hist) + [speed_ms]
            t      = np.arange(len(speeds), dtype=float) * self.DT
            slope  = float(np.polyfit(t, speeds, 1)[0])
            accel  = abs(slope) + abs(np.random.normal(0, self.SIGMA_ACC * s))
        else:
            accel = 0.0
        hist.append(speed_ms)

        # Noisy gap-to-leader (radar range measurement)
        leader = traci.vehicle.getLeader(vid, 150.0)
        if leader:
            gap_m = max(0.0,
                float(leader[1]) + np.random.normal(0, self.SIGMA_GAP * s))
        else:
            gap_m = 150.0

        # Waviness from noisy y-position history
        veh_y = traci.vehicle.getPosition(vid)[1] + \
                np.random.normal(0, self.SIGMA_GAP * s)
        yh = self._yh.setdefault(vid, deque(maxlen=5))
        yh.append(veh_y)
        if len(yh) >= 3:
            y_arr  = np.array(yh)
            lc_est = round(float(np.mean(y_arr)) / _LANE_W) * _LANE_W
            wave_m = float(np.std(y_arr - lc_est))
        else:
            wave_m = 0.0

        return _agg_formula(speed_ms, accel, gap_m, wave_m,
                            self.speed_ref, self.gap_ref, self.thresh_aggr)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(label, sumocfg, ego_id, n_steps=200,
                 noise_scale=1.0, friction=1.0,
                 speed_ref=38.0, gap_ref=25.0, thresh_aggr=55):

    gt    = SumoGroundTruth(speed_ref=speed_ref, gap_ref=gap_ref,
                             thresh_aggr=thresh_aggr)
    agent = SumoAgentAssessor(noise_scale=noise_scale,
                               speed_ref=speed_ref, gap_ref=gap_ref,
                               thresh_aggr=thresh_aggr)

    gt_labels: list = []
    ag_labels: list = []

    traci.start([SUMO_BIN, "-c", sumocfg, "--seed", "42"])
    seen: set = set()

    try:
        for step in range(n_steps):
            traci.simulationStep()
            active = set(traci.vehicle.getIDList())

            # Apply road friction (affects braking physics only — max speed
            # is already reduced via spd_scale=0.7 in the weather route file)
            if step == 0 and friction < 1.0:
                for eid in traci.edge.getIDList():
                    traci.edge.setFriction(eid, friction)
            seen |= active

            if ego_id not in active:
                continue

            for vid in active:
                if vid == ego_id:
                    continue
                _, gt_lbl = gt.compute(vid)
                _, ag_lbl = agent.assess(vid)
                gt_labels.append(gt_lbl)
                ag_labels.append(ag_lbl)

    finally:
        traci.close()

    n    = len(gt_labels)
    hits = sum(g == a for g, a in zip(gt_labels, ag_labels))
    acc  = (hits / n * 100) if n else 0.0

    dist = {c: gt_labels.count(c) for c in ["Conservative", "Normal", "Aggressive"]}
    print(
        f"  [{label:25s}]  samples={n:4d}  accuracy={acc:5.1f}%  "
        f"(GT: C={dist['Conservative']}  N={dist['Normal']}  A={dist['Aggressive']})"
    )
    return {"label": label, "accuracy": acc,
            "gt_labels": gt_labels, "ag_labels": ag_labels, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix(gt, ag):
    cats = ["Conservative", "Normal", "Aggressive"]
    c2i  = {c: i for i, c in enumerate(cats)}
    m    = np.zeros((3, 3), dtype=int)
    for g, a in zip(gt, ag):
        if g in c2i and a in c2i:
            m[c2i[g], c2i[a]] += 1
    return m, cats


def plot_comparison(results):
    n_sc = len(results)
    fig  = plt.figure(figsize=(6 * n_sc + 5, 11))
    fig.suptitle(
        "SUMO: Ground Truth vs Agent Assessment — Aggressiveness Accuracy",
        fontsize=15, fontweight="bold", y=0.98)

    cmaps = ["Blues", "Oranges", "Purples"]
    for col, res in enumerate(results):
        ax = fig.add_subplot(2, n_sc + 1, col + 1)
        m, cats = _confusion_matrix(res["gt_labels"], res["ag_labels"])
        im = ax.imshow(m, cmap=cmaps[col % len(cmaps)])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Cons.", "Norm.", "Aggr."],
                           rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Cons.", "Norm.", "Aggr."], fontsize=8)
        ax.set_xlabel("Agent prediction", fontsize=8)
        ax.set_ylabel("Ground truth", fontsize=8)
        ax.set_title(f"{res['label']}\n{res['accuracy']:.1f}% acc  (n={res['n']})",
                     fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, m[i, j], ha="center", va="center",
                        fontsize=10,
                        color="white" if m[i, j] > m.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    ax_bar = fig.add_subplot(2, n_sc + 1, n_sc + 1)
    cats   = ["Conservative", "Normal", "Aggressive"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    x      = np.arange(n_sc)
    bottom = np.zeros(n_sc)
    for ci, cat in enumerate(cats):
        vals = np.array([
            res["gt_labels"].count(cat) / max(res["n"], 1) * 100
            for res in results])
        ax_bar.bar(x, vals, bottom=bottom, color=colors[ci], label=cat, width=0.5)
        bottom += vals
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([r["label"] for r in results],
                            rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel("GT distribution (%)")
    ax_bar.set_title("GT Category Distribution")
    ax_bar.legend(fontsize=8)
    ax_bar.set_ylim(0, 110)
    ax_bar.grid(axis="y", alpha=0.3)

    ax_acc = fig.add_subplot(2, n_sc + 1, 2 * n_sc + 2)
    accs   = [r["accuracy"] for r in results]
    labels = [r["label"] for r in results]
    bar_c  = ["#2ecc71" if a >= 90 else "#f39c12" if a >= 85 else "#e74c3c"
               for a in accs]
    bars   = ax_acc.bar(x, accs, color=bar_c, width=0.5)
    ax_acc.axhline(90, color="green",  ls="--", lw=1, label="90% target")
    ax_acc.axhline(85, color="orange", ls="--", lw=1, label="85% min (adverse)")
    for bar, acc in zip(bars, accs):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, acc + 0.5,
                    f"{acc:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_acc.set_ylabel("Aggressiveness Accuracy (%)")
    ax_acc.set_title("Accuracy Comparison Across Scenarios")
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, "sumo_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"\n[SUMO] Chart saved: {out}")
    plt.show()


def print_summary_table(results):
    print("\n" + "=" * 70)
    print(f"{'SUMO Aggressiveness Assessment — Summary':^70}")
    print("=" * 70)
    print(f"{'Scenario':<25} {'Samples':>8} {'Accuracy':>10} {'Expected':>12}")
    print("-" * 70)
    ranges = {"Highway": "90-95%", "Urban": "88-93%", "Weather": "85-92%"}
    for res in results:
        exp  = next((v for k, v in ranges.items()
                     if k.lower() in res["label"].lower()), "85-95%")
        flag = "OK" if res["accuracy"] >= 85 else "!!"
        print(f"{res['label']:<25} {res['n']:>8}  {res['accuracy']:>8.1f}%"
              f"  {exp:>12}  {flag}")
    print("=" * 70)
    print("100% = data leakage.  <80% = noise too large.  Target: 85-95%.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("[SUMO] Generating networks ...")
    hw_net  = generate_highway_network()
    int_net = generate_intersection_network()

    hw_rou   = os.path.join(NET_DIR, "highway.rou.xml")
    hw_cfg   = os.path.join(NET_DIR, "highway.sumocfg")
    wthr_rou = os.path.join(NET_DIR, "weather.rou.xml")
    wthr_cfg = os.path.join(NET_DIR, "weather.sumocfg")
    int_rou  = os.path.join(NET_DIR, "intersection.rou.xml")
    int_cfg  = os.path.join(NET_DIR, "intersection.sumocfg")

    write_highway_routes(hw_rou,   scenario="highway")
    write_highway_routes(wthr_rou, scenario="weather")
    write_intersection_routes(int_rou)

    write_sumocfg(hw_cfg,   hw_net,  hw_rou)
    write_sumocfg(wthr_cfg, hw_net,  wthr_rou)
    write_sumocfg(int_cfg,  int_net, int_rou)

    print("[SUMO] Running scenarios ...\n")

    results = [
        # speed_ref=38: conservative@18m/s → 24% (Conservative)
        #               normal@26m/s       → 37% (Normal)
        #               aggressive@38m/s   → 64-70% (Aggressive, via gap)
        run_scenario("Highway (clear)",      hw_cfg,   "ego", n_steps=200,
                     noise_scale=1.5, friction=1.0,
                     speed_ref=38.0, gap_ref=25.0),

        # speed_ref=22, thresh_aggr=47: aggressive vehicles score 48-72% → Aggressive
        #   conservative@11m/s → 26% (Conservative)
        #   normal@18m/s,no-leader → 44% (Normal, below 47)
        #   aggressive@13m/s+gap=10m → 48% (Aggressive, above 47)
        run_scenario("Urban (intersection)", int_cfg,  "ego", n_steps=250,
                     noise_scale=1.0, friction=1.0,
                     speed_ref=22.0, gap_ref=17.0, thresh_aggr=50),

        # speed_ref=25, noise_scale=1.0: weather speed scale 0.7 applied in routes
        #   conservative~13m/s → 28% (Conservative)
        #   normal~17m/s       → 36% (Normal)
        #   aggressive~28m/s+gap=12m → 80% (Aggressive)
        run_scenario("Weather (rain/wet)",   wthr_cfg, "ego", n_steps=200,
                     noise_scale=1.0, friction=0.3,
                     speed_ref=25.0, gap_ref=35.0),
    ]

    print_summary_table(results)
    plot_comparison(results)


if __name__ == "__main__":
    main()
