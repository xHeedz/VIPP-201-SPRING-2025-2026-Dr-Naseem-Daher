"""Raw index inputs of one SUMO vehicle, measured the same way as UAH and NGSIM."""


def npc_features(traci, vid, lookahead_m=50.0):
    """(speed_kmh, accel_ms2, gap_m, wave_m) for vehicle vid; gap_m = 0 means no car ahead.

    speed: getSpeed in m/s, x 3.6.
    accel: getAcceleration, SUMO's accel over the last simulation step (independent of
           how often the caller samples).
    gap:   bumper to bumper gap to the leader in the same lane. getLeader's distance
           excludes the follower's minGap, so minGap is added back.
    wave:  |offset from the lane centre| (getLateralLanePosition), not travel distance.
    """
    speed_kmh = traci.vehicle.getSpeed(vid) * 3.6
    accel_ms2 = float(traci.vehicle.getAcceleration(vid))
    leader = traci.vehicle.getLeader(vid, lookahead_m)
    gap_m = float(leader[1] + traci.vehicle.getMinGap(vid)) if leader and leader[0] else 0.0
    wave_m = abs(float(traci.vehicle.getLateralLanePosition(vid)))
    return speed_kmh, accel_ms2, gap_m, wave_m
