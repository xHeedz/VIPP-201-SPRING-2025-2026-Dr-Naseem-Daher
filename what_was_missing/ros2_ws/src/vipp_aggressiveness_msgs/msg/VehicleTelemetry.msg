# Raw per-vehicle kinematics, one message per vehicle per simulation step.
# speed m/s, accel m/s^2, prox = gap to leader in m, wave = lateral peak-to-peak over 1 s in m,
# jerk m/s^3, slip = |v_actual - v_intended| / v_actual (kinematic proxy, 0 to 1)
float32 speed
float32 accel
float32 prox
float32 wave
float32 jerk
float32 slip
string vehicle_id
builtin_interfaces/Time stamp
