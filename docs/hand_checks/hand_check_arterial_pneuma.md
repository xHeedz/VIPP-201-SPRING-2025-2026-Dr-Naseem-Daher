# hand check: one NGSIM arterial row and one pNEUMA second

hand numbers: `scripts/hand_check_more.py` (plain python on the raw CSV rows); pipeline: `datasets/ngsim.py`
(`trajectories(planar=True)`, `per_second`) and `datasets/pneuma.py` (`per_second`).

## ngsim lankershim, vehicle 100065 (period 1, id 65), t = 29.0 s

frame 293, lane 31, direction 2, section 3, through movement, not in an intersection.

- speed: 2-D path (arterial). velocity from the 10 frame centred means of Local_X and Local_Y: vx -0.030, vy 3.710 m/s, |v| = **13.35 km/h**
- accel_1hz: 3 s mean of the per second speeds, previous and next second 3.5096 and 1.1047 m/s, difference / 2 s = **-1.202 m/s2**
- gap: Space_Headway 29.32 ft minus leader 100063 length 14.2 ft, x 0.3048 = **4.61 m**
- wave: smoothed Local_X 15.118 m, lane centre = median of 5531 smoothed rows in the same lane, direction, section and 10 m of road 15.342 m, |difference| = **0.224 m**
- features 0.0079, 0.2405, 0.8242, 0.1495, score **77.12**

pipeline: speed 13.35, accel -1.202, gap 4.61, wave 0.224, score **77.12**; difference 0.00e+00.

## pneuma, vehicle 19 (Car), second 37

- speed: pNEUMA speed at the whole seconds around it 20.27, 16.88, 13.92, 10.21, 7.22 km/h, 3 s mean = **13.67 km/h**
- accel: derivative of the 3 s mean speed = **-0.913 m/s2**
- heading from the positions 1 s before and after: -58.4 deg
- gap: nearest vehicle ahead in the cone (+-1.6 m lateral, heading within 30 deg) is 46 (Car, stopped: its heading is the last one while moving), 13.76 m ahead and 0.08 m to the side; minus half of both lengths = **9.26 m**
- wave: not measurable (no lane map), 0
- vehicles within 50 m: 9
- features 0.0083, 0.1826, 0.6640, 0.0000, score **57.19**

pipeline: speed 13.67, accel -0.913, gap 9.26, density 9, score **57.19**; difference 6.38e-12.
