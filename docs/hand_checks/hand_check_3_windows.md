# hand check: aggressiveness index on 3 real windows

reference: `model/aggressiveness_model.py`. hand numbers: `scripts/hand_check.py` (plain python on the raw
files, no pipeline code). pipeline numbers: `datasets/uah.py` `windows` and `datasets/ngsim.py` `trajectories`.

formula: AI = min(100 (0.5 n_s^2 + 0.2 n_a + 0.8 n_p^2 + 0.4 n_w), 100); n_s = min(v_kmh/150, 1),
n_a = min(|a|/5, 1), n_p = 1 - gap/50 if 0 < gap <= 50 else 0, n_w = min(|wave|/1.5, 1);
labels: < 29 conservative, < 42 normal, else aggressive (fitted, scripts/fit_cutoffs.py).

## uah d1 normal motorway, window t0 = 306.88 s, 10 s

trip `20151111123124-25km-D1-NORMAL-MOTORWAY`. one row per GPS second; the window score is the score of the mean features
(equal to the mean of the per second scores while no second reaches the cap of 100).

- speed: RAW_GPS col 1 (km/h), / 3.6, 3 s centred mean, x 3.6 back to km/h, / 150, squared
- accel: derivative of the smoothed speed at the GPS timestamps (np.gradient, second order for uneven dt;
  for equal dt it is (v_next - v_prev) / 2 dt), clip to +-9, |a| / 5
- gap: PROC_VEHICLE_DETECTION col 1 (m, to the vehicle ahead seen by the phone camera), first row at or
  after the GPS time and within 1.5 s; <= 0 means no vehicle; 1 - gap/50, squared
- wave: PROC_LANE_DETECTION col 1 (m, car position from the lane centre), rows with road width > 0,
  |offset| <= 2 m and estimator state 2 (detected), first row at or after the GPS time and within 1 s; |x| / 1.5

| t (s) | gps prev / now / next (km/h) | dt l / r (s) | speed (km/h) | accel (m/s2) | gap (m) | wave (m) | n_s^2 | n_a | n_p^2 | n_w | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 306.88 | 106.5 / 106.9 / 107.5 | 0.95 / 1.01 | 106.97 | 0.062 | 30.07 | 0.040 | 0.5085 | 0.0124 | 0.1589 | 0.0267 | 39.45 |
| 307.89 | 106.9 / 107.5 / 107.9 | 1.01 / 0.99 | 107.43 | 0.092 | 30.39 | 0.030 | 0.5130 | 0.0184 | 0.1538 | 0.0200 | 39.12 |
| 308.88 | 107.5 / 107.9 / 107.5 | 0.99 / 1.08 | 107.63 | 0.029 | 29.97 | 0.000 | 0.5149 | 0.0059 | 0.1605 | 0.0000 | 38.70 |
| 309.96 | 107.9 / 107.5 / 107.5 | 1.08 / 0.92 | 107.63 | -0.049 | 30.03 | 0.109 | 0.5149 | 0.0098 | 0.1595 | 0.0727 | 41.61 |
| 310.88 | 107.5 / 107.5 / 107.0 | 0.92 / 1.07 | 107.33 | -0.113 | 28.75 | 0.020 | 0.5120 | 0.0225 | 0.1806 | 0.0133 | 41.04 |
| 311.95 | 107.5 / 107.0 / 105.9 | 1.07 / 0.93 | 106.80 | -0.139 | 28.47 | 0.040 | 0.5069 | 0.0278 | 0.1854 | 0.0267 | 41.80 |
| 312.88 | 107.0 / 105.9 / 106.1 | 0.93 / 1.00 | 106.33 | -0.095 | 28.87 | 0.099 | 0.5025 | 0.0189 | 0.1786 | 0.0660 | 42.43 |
| 313.88 | 105.9 / 106.1 / 106.5 | 1.00 / 1.01 | 106.17 | -0.069 | 27.65 | 0.050 | 0.5009 | 0.0138 | 0.1998 | 0.0333 | 42.64 |
| 314.89 | 106.1 / 106.5 / 104.9 | 1.01 / 1.06 | 105.83 | -0.107 | 27.08 | 0.200 | 0.4978 | 0.0213 | 0.2101 | 0.1333 | 47.46 |
| 315.95 | 106.5 / 104.9 / 104.7 | 1.06 / 0.98 | 105.37 | -0.157 | 27.04 | 0.255 | 0.4934 | 0.0314 | 0.2109 | 0.1700 | 48.97 |

mean features: n_s^2 0.5065, n_a 0.0182, n_p^2 0.1798, n_w 0.0562
weighted: 0.5 x 0.5065 = 0.2532; 0.2 x 0.0182 = 0.0036; 0.8 x 0.1798 = 0.1439; 0.4 x 0.0562 = 0.0225
sum 0.4232, x 100 = **42.32**, label **aggressive**

pipeline: features 0.5065, 0.0182, 0.1798, 0.0562, score **42.32**, label aggressive. difference 7.11e-15.

## uah d1 aggressive motorway, window t0 = 688.94 s, 10 s

trip `20151111125233-24km-D1-AGGRESSIVE-MOTORWAY`. one row per GPS second; the window score is the score of the mean features
(equal to the mean of the per second scores while no second reaches the cap of 100).

- speed: RAW_GPS col 1 (km/h), / 3.6, 3 s centred mean, x 3.6 back to km/h, / 150, squared
- accel: derivative of the smoothed speed at the GPS timestamps (np.gradient, second order for uneven dt;
  for equal dt it is (v_next - v_prev) / 2 dt), clip to +-9, |a| / 5
- gap: PROC_VEHICLE_DETECTION col 1 (m, to the vehicle ahead seen by the phone camera), first row at or
  after the GPS time and within 1.5 s; <= 0 means no vehicle; 1 - gap/50, squared
- wave: PROC_LANE_DETECTION col 1 (m, car position from the lane centre), rows with road width > 0,
  |offset| <= 2 m and estimator state 2 (detected), first row at or after the GPS time and within 1 s; |x| / 1.5

| t (s) | gps prev / now / next (km/h) | dt l / r (s) | speed (km/h) | accel (m/s2) | gap (m) | wave (m) | n_s^2 | n_a | n_p^2 | n_w | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 688.97 | 131.0 / 126.0 / 126.0 | 1.02 / 0.95 | 127.67 | -0.320 | 13.52 | 0.061 | 0.7244 | 0.0639 | 0.5323 | 0.0407 | 81.71 |
| 689.92 | 126.0 / 126.0 / 125.1 | 0.95 / 1.10 | 125.70 | -0.344 | 13.15 | 0.101 | 0.7022 | 0.0687 | 0.5432 | 0.0673 | 82.63 |
| 691.02 | 126.0 / 125.1 / 125.1 | 1.10 / 0.86 | 125.40 | -0.106 | 12.41 | 0.030 | 0.6989 | 0.0212 | 0.5652 | 0.0200 | 81.38 |
| 691.88 | 125.1 / 125.1 / 124.8 | 0.86 / 1.08 | 125.00 | -0.087 | 12.27 | 0.040 | 0.6944 | 0.0174 | 0.5694 | 0.0267 | 81.69 |
| 692.96 | 125.1 / 124.8 / 124.7 | 1.08 / 1.05 | 124.87 | -0.035 | 12.02 | 0.010 | 0.6930 | 0.0070 | 0.5770 | 0.0067 | 81.21 |
| 694.01 | 124.8 / 124.7 / 124.7 | 1.05 / 0.90 | 124.73 | 0.072 | 11.94 | 0.199 | 0.6915 | 0.0145 | 0.5794 | 0.1327 | 86.52 |
| 694.91 | 124.7 / 124.7 / 126.4 | 0.90 / 1.07 | 125.27 | 0.157 | 11.71 | 0.535 | 0.6974 | 0.0313 | 0.5864 | 0.3567 | 96.68 |
| 695.98 | 124.7 / 126.4 / 126.4 | 1.07 / 0.94 | 125.83 | -0.036 | 11.58 | 0.679 | 0.7037 | 0.0072 | 0.5904 | 0.4527 | 100.00 |
| 696.92 | 126.4 / 126.4 / 122.7 | 0.94 / 1.05 | 125.17 | -0.225 | 11.43 | 0.181 | 0.6963 | 0.0449 | 0.5951 | 0.1207 | 88.15 |
| 697.97 | 126.4 / 122.7 / 123.5 | 1.05 / 0.92 | 124.20 | -0.302 | 11.49 | 0.127 | 0.6856 | 0.0604 | 0.5932 | 0.0847 | 86.33 |
| 698.89 | 122.7 / 123.5 / 123.0 | 0.92 / 1.04 | 123.07 | -0.190 | 11.69 | 0.270 | 0.6731 | 0.0380 | 0.5871 | 0.1800 | 88.58 |

mean features: n_s^2 0.6964, n_a 0.0340, n_p^2 0.5744, n_w 0.1353
weighted: 0.5 x 0.6964 = 0.3482; 0.2 x 0.0340 = 0.0068; 0.8 x 0.5744 = 0.4595; 0.4 x 0.1353 = 0.0541
sum 0.8687, x 100 = **86.87**, label **aggressive**

pipeline: features 0.6964, 0.0340, 0.5744, 0.1353, score **86.87**, label aggressive. difference 0.00e+00.

## ngsim us-101, car 983 at t = 259.0 s (frame 2598, lane 2)

one 10 Hz row; the vehicle score in `score_ngsim.py` is the mean of these row scores.

- speed: Local_Y (ft) x 0.3048, 10 frame centred mean, gradient over t. Local_Y rows i-5..i+5: 814.991, 818.104, 821.235, 824.353, 827.424, 830.46, 833.508, 836.606, 839.73, 842.819, 845.839 ft. smoothed position at i-1 and i+1: 251.714, 253.596 m, dt 0.10 / 0.10 s, speed 9.4124 m/s = **33.88 km/h**
- accel (accel_1hz, measured like UAH): per second mean speeds around this second 8.277, 9.367, 9.476, 10.506, 11.706 m/s; 3 s centred means of the previous and next second 9.0397 and 10.5623 m/s; (10.5623 - 9.0397) / 2 s = **0.761 m/s2**
- gap: Space_Headway 68.33 ft (front to front) minus leader 979 length 16.5 ft, x 0.3048 = **15.80 m**
- wave: smoothed Local_X 5.353 m, lane centre (median of 69379 smoothed rows in the lane) 5.708 m, |difference| = **0.355 m**

n_s = 33.88/150 = 0.2259, squared 0.0510; n_a = 0.761/5 = 0.1523; n_p = 1 - 15.80/50 = 0.6840, squared 0.4679; n_w = 0.355/1.5 = 0.2368
weighted: 0.0255 + 0.0305 + 0.3743 + 0.0947 = 0.5250, x 100 = **52.50**, label **aggressive**

pipeline: speed 33.88 km/h, accel 0.761, gap 15.80 m, wave 0.355 m, score **52.50**, label aggressive. difference 7.11e-14.

what drives this score: proximity 37.4 of 52.5 points, from a 15.8 m gap at 34 km/h (time headway 1.7 s, ordinary for dense traffic).
