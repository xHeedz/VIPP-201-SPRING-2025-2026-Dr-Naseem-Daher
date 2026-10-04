"""
Ellipse shapes for drawing vehicles, instead of points.

Every vehicle gets two ellipses, both lying along the direction of travel:

  body       the vehicle's footprint: half-length a = L / 2, half-width b = W / 2

  influence  the road the driver claims around itself. It stretches forward with
             speed and grows with the Aggressiveness Index (AI, 0 to 100):
                 T = T_BASE + T_AI * AI / 100                 seconds of travel claimed
                 a = L / 2 + speed * T                         half-length
                 b = W / 2 + B_BASE + B_AI * AI / 100          half-width
             and its centre is moved forward by FORWARD_SHIFT * speed * T,
             so most of it lies ahead of the vehicle.

Worked example: L = 4.5 m, W = 1.8 m, speed 25 m/s, AI 80
    T = 0.5 + 1.5 * 0.8 = 1.7 s,  a = 2.25 + 25 * 1.7 = 44.75 m,  b = 0.9 + 0.5 + 0.8 = 2.2 m
The same vehicle at AI 20 gets T = 0.8 s and a = 22.25 m.

The colour follows the three categories: red aggressive (AI at or above the
aggressive threshold), green conservative (below 29/42 of it, the reference cut offs
29 and 42), yellow normal in between. The trained agent learns its own threshold,
so both boundaries move with it.

Everything here is plain math on numbers, with no SUMO inside, so the shapes can
be tested on their own and reused by any script (SUMO, plots, ROS2).
Coordinates are metres; heading is the direction of travel in radians,
0 meaning along +x.
"""
import math

from model.aggressiveness_model import THRESHOLDS

N_POINTS = 24            # points used to trace each ellipse

T_BASE = 0.5             # s   claimed travel time at AI 0
T_AI = 1.5               # s   extra claimed travel time at AI 100
B_BASE = 0.5             # m   extra half-width at AI 0
B_AI = 1.0               # m   extra half-width at AI 100
FORWARD_SHIFT = 0.5      # fraction of the speed stretch the centre moves forward

CONSERVATIVE_SHARE = THRESHOLDS[0] / THRESHOLDS[1]     # conservative below this share of the aggressive threshold (29 / 42)

GREEN = (46, 160, 67)
YELLOW = (230, 180, 0)
RED = (210, 40, 40)
PURPLE = (140, 60, 200)     # outline for vehicles disturbing the traffic behind them


def ellipse_points(cx, cy, a, b, heading=0.0, n=N_POINTS):
    """n points (x, y) on the ellipse with centre (cx, cy), half-axes a (along heading) and b."""
    ch, sh = math.cos(heading), math.sin(heading)
    pts = []
    for k in range(n):
        th = 2.0 * math.pi * k / n
        u, v = a * math.cos(th), b * math.sin(th)          # ellipse in its own frame
        pts.append((cx + u * ch - v * sh, cy + u * sh + v * ch))   # rotate to the heading, then move
    return pts


def body_ellipse(x_front, y, length, width, heading=0.0):
    """Footprint of a vehicle whose front bumper centre is at (x_front, y)."""
    cx = x_front - 0.5 * length * math.cos(heading)
    cy = y - 0.5 * length * math.sin(heading)
    return ellipse_points(cx, cy, 0.5 * length, 0.5 * width, heading)


def influence_axes(length, width, speed, ai):
    """(a, b, forward stretch) of the influence ellipse. speed in m/s, ai in 0 to 100."""
    level = min(max(ai, 0.0), 100.0) / 100.0
    stretch = max(speed, 0.0) * (T_BASE + T_AI * level)
    a = 0.5 * length + stretch
    b = 0.5 * width + B_BASE + B_AI * level
    return a, b, stretch


def influence_ellipse(x_front, y, length, width, speed, ai, heading=0.0):
    """Influence zone of a vehicle whose front bumper centre is at (x_front, y)."""
    a, b, stretch = influence_axes(length, width, speed, ai)
    shift = -0.5 * length + FORWARD_SHIFT * stretch        # from the front bumper to the zone centre
    cx = x_front + shift * math.cos(heading)
    cy = y + shift * math.sin(heading)
    return ellipse_points(cx, cy, a, b, heading)


def category(ai, aggressive_from=THRESHOLDS[1]):
    if ai >= aggressive_from:
        return "aggressive"
    if ai < CONSERVATIVE_SHARE * aggressive_from:
        return "conservative"
    return "normal"


def color_for(ai, aggressive_from=THRESHOLDS[1], alpha=255):
    """RGBA colour for a score: green, yellow or red."""
    rgb = {"conservative": GREEN, "normal": YELLOW, "aggressive": RED}[category(ai, aggressive_from)]
    return rgb + (alpha,)


def color_of(cat, alpha=255):
    """RGBA colour for a category name: conservative, normal or aggressive."""
    return {"conservative": GREEN, "normal": YELLOW, "aggressive": RED}[cat] + (alpha,)
