"""
Unified reward function for all three driving scenarios.

One function handles Highway, Urban/Intersection, and Weather/Friction so the
agent cannot overfit to scenario-specific reward signals.

Actions
-------
highway / weather : 0=LaneLeft  1=Idle  2=LaneRight  3=Faster  4=Slower
urban             : 0=Brake     1=Idle  2=Accelerate

Context dict keys
-----------------
highway / weather:
    front_dist             float   metres to nearest forward vehicle  (default 50)
    nearby_aggressiveness  float   max aggressiveness in neighbourhood [0,1]  (default 0)

urban:
    front_dist             float
    cross_left             float   distance to nearest left-crossing vehicle
    cross_right            float   distance to nearest right-crossing vehicle
    left_aggressiveness    float   [0,1]
    right_aggressiveness   float   [0,1]
"""


def unified_reward(
    scenario:       str,
    ego_speed:      float,
    action_idx:     int,
    crashed:        bool,
    context:        dict,
    weather_factor: float = 1.0,
) -> float:
    """
    Parameters
    ----------
    scenario       : 'highway' | 'urban' | 'weather'
    ego_speed      : ego vehicle speed in m/s
    action_idx     : discrete action taken
    crashed        : True if a collision occurred this step
    context        : scenario-specific dict (see module docstring)
    weather_factor : 1.0 = clear road, 0.7 = rain, 0.5 = heavy snow / ice
                     Scales target speed and safe-following-distance thresholds.
    """
    if crashed:
        return -50.0

    reward = 1.0    # base survival bonus shared by every scenario

    # ── HIGHWAY ──────────────────────────────────────────────────────────────
    if scenario == "highway":
        front_dist = context.get("front_dist", 50.0)
        near_aggr  = context.get("nearby_aggressiveness", 0.0)

        # Speed incentive — target ≈ 90 km/h on a clear road
        target_spd = 25.0 * weather_factor
        if ego_speed >= target_spd * 0.8:
            reward += 1.5
        elif ego_speed < target_spd * 0.5:
            reward -= 1.0

        # Following distance — widen threshold when neighbours are aggressive
        safe_dist = (12.0 + 8.0 * near_aggr) / weather_factor
        if front_dist < safe_dist:
            reward -= 2.0 + 2.0 * near_aggr

        # Lane-change cost (small: allowed when blocked, discouraged otherwise)
        if action_idx in [0, 2]:
            reward -= 0.5

    # ── URBAN / INTERSECTION ─────────────────────────────────────────────────
    elif scenario == "urban":
        front_dist  = context.get("front_dist",  50.0)
        cross_left  = context.get("cross_left",  50.0)
        cross_right = context.get("cross_right", 50.0)
        l_aggr      = context.get("left_aggressiveness",  0.0)
        r_aggr      = context.get("right_aggressiveness", 0.0)

        # Loitering tax — penalise waiting when the intersection is clear
        if ego_speed < 3.0 and cross_left > 20.0 and cross_right > 20.0 and front_dist > 15.0:
            reward -= 3.0

        # Aggressiveness-scaled threat radius
        l_thresh = 8.0 + 7.0 * l_aggr
        r_thresh = 8.0 + 7.0 * r_aggr
        under_threat = cross_left < l_thresh or cross_right < r_thresh

        if under_threat:
            threat = max(
                l_aggr if cross_left  < l_thresh else 0.0,
                r_aggr if cross_right < r_thresh else 0.0,
            )
            if action_idx == 0:      # braking — correct response to threat
                reward += (3.0 + 2.0 * threat) if ego_speed > 1.5 else -1.5
            elif action_idx == 2:    # accelerating into threat — dangerous
                reward -= 5.0 + 5.0 * threat
        else:
            # Road clear — reward making progress
            if front_dist > 20.0:
                if action_idx == 2:
                    reward += 1.5
                reward += ego_speed / 10.0

    # ── WEATHER / FRICTION ───────────────────────────────────────────────────
    elif scenario == "weather":
        front_dist = context.get("front_dist", 50.0)
        near_aggr  = context.get("nearby_aggressiveness", 0.0)

        # Target speed is substantially reduced in poor conditions
        target_spd = 15.0 * weather_factor
        if ego_speed > target_spd * 1.3:
            reward -= 3.0           # penalty for excessive speed in bad weather
        elif ego_speed > target_spd * 0.6:
            reward += 1.5           # reward appropriate cautious speed

        # Much larger following distance needed on slippery roads
        safe_dist = 20.0 / max(weather_factor, 0.3)
        if front_dist < safe_dist:
            reward -= 3.0 + 2.0 * near_aggr

        # Any abrupt lateral input is penalised (risk of skidding)
        if action_idx in [0, 2]:
            reward -= 1.0

    return float(reward)
