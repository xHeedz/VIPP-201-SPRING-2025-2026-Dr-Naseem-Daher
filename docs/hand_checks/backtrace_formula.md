# backtrace: reference formula, from raw values to the label

formula: score = 100 x (sum of weight x term over the known terms) / (sum of the known weights). weights speed 0.25, accel² 0.35, gap 0.30, lane 0.10. terms: speed = km/h / 150; accel² = (|a| / 5)²; gap = mix of (1 - metres / 50)² and (1 - headway / 3 s)²; lane = offset / 1.5 m. a window term is the mean of its per second terms.

## UAH, aggressive trip (median window)

window: `20151221120051-26km-D6-AGGRESSIVE-MOTORWAY`, t0 370 s, environment highway (gap mix: 0.5 metres, 0.5 seconds), true label aggressive

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 370 | 110.7 | 0.03 | 20.7 | yes | 0.31 | 0.738 | 0.0000 | 0.472 | 0.208 |
| 371 | 110.9 | -0.01 | 19.8 | yes | 0.02 | 0.739 | 0.0000 | 0.492 | 0.013 |
| 372 | 110.6 | -0.09 | 19.8 | yes | 0.17 | 0.737 | 0.0003 | 0.490 | 0.114 |
| 373 | 110.2 | -0.16 | 19.8 | yes | 0.20 | 0.735 | 0.0010 | 0.491 | 0.134 |
| 374 | 109.5 | -0.16 | 19.6 | yes | 0.08 | 0.730 | 0.0010 | 0.493 | 0.054 |
| 375 | 109.1 | -0.08 | 19.7 | yes | 0.05 | 0.727 | 0.0003 | 0.491 | 0.033 |
| 376 | 108.9 | -0.03 | 19.8 | yes | 0.07 | 0.726 | 0.0000 | 0.488 | 0.047 |
| 377 | 108.9 | 0.00 | 20.6 | yes | 0.04 | 0.726 | 0.0000 | 0.471 | 0.027 |
| 378 | 108.9 | 0.01 | 21.3 | yes | 0.11 | 0.726 | 0.0000 | 0.458 | 0.074 |
| 379 | 109.0 | 0.06 | 21.8 | yes | 0.06 | 0.726 | 0.0001 | 0.448 | 0.040 |
| 380 | 109.3 | 0.13 | 22.0 | yes | 0.03 | 0.728 | 0.0007 | 0.444 | 0.020 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.7308 | 0.25 | 18.3 |
| accel2 | 0.0003 | 0.35 | 0.0 |
| prox2 | 0.4762 | 0.30 | 14.3 |
| wave | 0.0695 | 0.10 | 0.7 |

known weights add up to 1.00. score = 33.3, label **aggressive** (cut offs 19.0 / 27.5). largest part: speed (18.3 of 33.3 points).

## UAH, normal trip (median window)

window: `20151120162105-17km-D2-NORMAL2-SECONDARY`, t0 390 s, environment urban (gap mix: 0.5 metres, 0.5 seconds), true label normal

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 390 | 90.7 | -0.31 | unknown | unknown | 0.13 | 0.605 | 0.0039 | unknown | 0.088 |
| 391 | 89.6 | -0.25 | unknown | unknown | 0.21 | 0.598 | 0.0025 | unknown | 0.142 |
| 392 | 88.9 | -0.11 | unknown | unknown | 0.25 | 0.593 | 0.0005 | unknown | 0.169 |
| 393 | 88.9 | 0.07 | unknown | unknown | 0.28 | 0.592 | 0.0002 | unknown | 0.189 |
| 394 | 89.5 | 0.20 | unknown | unknown | 0.31 | 0.596 | 0.0016 | unknown | 0.209 |
| 395 | 90.3 | 0.24 | unknown | unknown | 0.29 | 0.602 | 0.0022 | unknown | 0.196 |
| 396 | 91.1 | 0.18 | unknown | unknown | 0.34 | 0.608 | 0.0014 | unknown | 0.230 |
| 397 | 91.6 | 0.08 | unknown | unknown | 0.33 | 0.611 | 0.0002 | unknown | 0.223 |
| 398 | 91.7 | -0.07 | unknown | unknown | 0.32 | 0.611 | 0.0002 | unknown | 0.216 |
| 399 | 91.1 | -0.14 | unknown | unknown | 0.21 | 0.607 | 0.0008 | unknown | 0.142 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.6022 | 0.25 | 21.5 |
| accel2 | 0.0014 | 0.35 | 0.1 |
| prox2 | unknown | 0.30 | unknown |
| wave | 0.1804 | 0.10 | 2.6 |

known weights add up to 0.70. score = 24.2, label **normal** (cut offs 19.0 / 27.5). largest part: speed (21.5 of 24.2 points).

## simulated aggressive car

window: `merge_medium_s1/aggressive_main.66`, t0 573 s, environment highway (gap mix: 0.5 metres, 0.5 seconds), true label aggressive

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 573 | 35.1 | -1.06 | 6.5 | yes | 1.11 | 0.234 | 0.0453 | 0.680 | 0.741 |
| 574 | 30.9 | -0.97 | 5.6 | yes | 1.11 | 0.206 | 0.0374 | 0.699 | 0.741 |
| 575 | 28.1 | -0.77 | 5.5 | yes | 1.11 | 0.188 | 0.0238 | 0.688 | 0.741 |
| 576 | 25.4 | -1.16 | 5.5 | yes | 1.11 | 0.169 | 0.0541 | 0.667 | 0.741 |
| 577 | 19.8 | -1.79 | 4.4 | yes | 1.11 | 0.132 | 0.1281 | 0.683 | 0.741 |
| 578 | 12.5 | -1.77 | 3.1 | yes | 1.11 | 0.083 | 0.1251 | 0.684 | 0.741 |
| 579 | 7.0 | -0.95 | 2.9 | yes | 0.75 | 0.047 | 0.0363 | 0.572 | 0.501 |
| 580 | 5.6 | -0.02 | 3.2 | yes | 0.05 | 0.037 | 0.0000 | 0.491 | 0.034 |
| 581 | 6.9 | 0.40 | 3.5 | yes | 0.10 | 0.046 | 0.0063 | 0.505 | 0.068 |
| 582 | 8.5 | 0.51 | 3.8 | yes | 0.10 | 0.057 | 0.0105 | 0.534 | 0.067 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.1198 | 0.25 | 3.0 |
| accel2 | 0.0467 | 0.35 | 1.6 |
| prox2 | 0.6203 | 0.30 | 18.6 |
| wave | 0.5116 | 0.10 | 5.1 |

known weights add up to 1.00. score = 28.4, label **aggressive** (cut offs 19.0 / 27.5). largest part: prox2 (18.6 of 28.4 points).

## simulated normal car

window: `jam_jam_s9/normal_r0.269`, t0 669 s, environment highway (gap mix: 0.5 metres, 0.5 seconds), true label normal

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 669 | 112.1 | -0.09 | 81.1 | yes | 0.10 | 0.748 | 0.0003 | 0.009 | 0.067 |
| 670 | 111.7 | -0.16 | 76.3 | yes | 0.10 | 0.745 | 0.0010 | 0.016 | 0.067 |
| 671 | 111.0 | -0.21 | 71.6 | yes | 0.07 | 0.740 | 0.0017 | 0.025 | 0.046 |
| 672 | 110.2 | -0.24 | 67.2 | yes | 0.10 | 0.735 | 0.0023 | 0.036 | 0.066 |
| 673 | 109.3 | -0.27 | 63.1 | yes | 0.09 | 0.729 | 0.0029 | 0.047 | 0.061 |
| 674 | 108.3 | -0.28 | 59.2 | yes | 0.07 | 0.722 | 0.0032 | 0.059 | 0.048 |
| 675 | 107.2 | -0.29 | 55.7 | yes | 0.08 | 0.715 | 0.0034 | 0.071 | 0.050 |
| 676 | 106.2 | -0.29 | 52.5 | yes | 0.07 | 0.708 | 0.0034 | 0.083 | 0.044 |
| 677 | 105.1 | -0.28 | 49.6 | yes | 0.10 | 0.701 | 0.0032 | 0.094 | 0.066 |
| 678 | 104.1 | -0.27 | 46.9 | yes | 0.09 | 0.694 | 0.0030 | 0.107 | 0.061 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.7236 | 0.25 | 18.1 |
| accel2 | 0.0024 | 0.35 | 0.1 |
| prox2 | 0.0548 | 0.30 | 1.6 |
| wave | 0.0576 | 0.10 | 0.6 |

known weights add up to 1.00. score = 20.4, label **normal** (cut offs 19.0 / 27.5). largest part: speed (18.1 of 20.4 points).

## simulated conservative car

window: `merge_medium_s1/conservative_main.90`, t0 696 s, environment highway (gap mix: 0.5 metres, 0.5 seconds), true label conservative

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 696 | 93.1 | -0.03 | 103.8 | yes | 0.03 | 0.621 | 0.0000 | 0.000 | 0.018 |
| 697 | 92.9 | -0.04 | 102.5 | yes | 0.02 | 0.620 | 0.0001 | 0.000 | 0.012 |
| 698 | 92.8 | -0.06 | 100.9 | yes | 0.00 | 0.618 | 0.0001 | 0.000 | 0.002 |
| 699 | 92.5 | -0.07 | 99.1 | yes | 0.03 | 0.617 | 0.0002 | 0.000 | 0.022 |
| 700 | 92.2 | -0.08 | 97.1 | yes | 0.03 | 0.615 | 0.0002 | 0.000 | 0.021 |
| 701 | 92.0 | -0.08 | 95.1 | yes | 0.04 | 0.613 | 0.0002 | 0.000 | 0.026 |
| 702 | 91.7 | -0.08 | 93.1 | yes | 0.13 | 0.611 | 0.0002 | 0.000 | 0.085 |
| 703 | 91.4 | -0.08 | 91.1 | yes | 0.11 | 0.609 | 0.0003 | 0.000 | 0.073 |
| 704 | 91.1 | -0.10 | 89.1 | yes | 0.10 | 0.607 | 0.0004 | 0.000 | 0.065 |
| 705 | 90.7 | -0.12 | 86.8 | yes | 0.13 | 0.605 | 0.0006 | 0.000 | 0.089 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.6136 | 0.25 | 15.3 |
| accel2 | 0.0002 | 0.35 | 0.0 |
| prox2 | 0.0000 | 0.30 | 0.0 |
| wave | 0.0413 | 0.10 | 0.4 |

known weights add up to 1.00. score = 15.8, label **conservative** (cut offs 19.0 / 27.5). largest part: speed (15.3 of 15.8 points).

## highD, German motorway (median window)

window: `36_1370`, t0 652 s, environment highway (gap mix: 0.5 metres, 0.5 seconds), true label unlabelled

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 652 | 112.8 | -0.13 | 82.1 | yes | 0.09 | 0.752 | 0.0006 | 0.008 | 0.060 |
| 653 | 112.3 | -0.19 | 82.8 | yes | 0.35 | 0.749 | 0.0014 | 0.007 | 0.233 |
| 654 | 111.4 | -0.24 | 83.7 | yes | 0.79 | 0.743 | 0.0023 | 0.005 | 0.527 |
| 655 | 110.6 | -0.22 | 84.7 | yes | 1.35 | 0.737 | 0.0019 | 0.003 | 0.900 |
| 656 | 109.9 | -0.17 | 112.2 | yes | 1.75 | 0.732 | 0.0012 | 0.000 | 1.000 |
| 657 | 109.3 | -0.14 | 108.3 | yes | 1.09 | 0.729 | 0.0008 | 0.000 | 0.727 |
| 658 | 108.9 | -0.17 | 104.4 | yes | 0.52 | 0.726 | 0.0012 | 0.000 | 0.347 |
| 659 | 108.1 | -0.40 | 99.5 | yes | 0.13 | 0.721 | 0.0063 | 0.000 | 0.087 |
| 660 | 106.0 | -0.91 | 92.6 | yes | 0.00 | 0.707 | 0.0333 | 0.000 | 0.000 |
| 661 | 101.5 | -1.50 | 84.2 | yes | 0.02 | 0.677 | 0.0898 | 0.000 | 0.013 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.7271 | 0.25 | 18.2 |
| accel2 | 0.0139 | 0.35 | 0.5 |
| prox2 | 0.0023 | 0.30 | 0.1 |
| wave | 0.3893 | 0.10 | 3.9 |

known weights add up to 1.00. score = 22.6, label **normal** (cut offs 19.0 / 27.5). largest part: speed (18.2 of 22.6 points).

## NGSIM US-101 (median window)

window: `3102`, t0 125 s, environment highway (gap mix: 0.5 metres, 0.5 seconds), true label unlabelled

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 125 | 16.7 | 0.86 | 21.9 | yes | 1.25 | 0.112 | 0.0299 | 0.157 | 0.834 |
| 126 | 34.0 | 1.15 | 23.3 | yes | 1.25 | 0.227 | 0.0524 | 0.159 | 0.832 |
| 127 | 37.5 | 1.00 | 24.1 | yes | 1.18 | 0.250 | 0.0397 | 0.159 | 0.787 |
| 128 | 39.2 | 0.29 | 23.0 | yes | 1.17 | 0.261 | 0.0033 | 0.189 | 0.777 |
| 129 | 41.1 | -0.02 | 21.2 | yes | 0.85 | 0.274 | 0.0000 | 0.239 | 0.567 |
| 130 | 40.5 | -0.04 | 18.7 | yes | 0.66 | 0.270 | 0.0001 | 0.295 | 0.442 |
| 131 | 38.5 | 0.14 | 17.3 | yes | 0.47 | 0.257 | 0.0008 | 0.320 | 0.314 |
| 132 | 40.0 | 0.31 | 16.8 | yes | 0.49 | 0.267 | 0.0039 | 0.344 | 0.328 |
| 133 | 41.9 | 0.27 | 16.0 | yes | 0.32 | 0.279 | 0.0028 | 0.378 | 0.215 |
| 134 | 42.9 | 0.18 | 15.1 | yes | 0.45 | 0.286 | 0.0012 | 0.410 | 0.299 |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.2483 | 0.25 | 6.2 |
| accel2 | 0.0134 | 0.35 | 0.5 |
| prox2 | 0.2651 | 0.30 | 8.0 |
| wave | 0.5396 | 0.10 | 5.4 |

known weights add up to 1.00. score = 20.0, label **normal** (cut offs 19.0 / 27.5). largest part: prox2 (8.0 of 20.0 points).

## pNEUMA, Athens (median window)

window: `301901`, t0 601 s, environment urban (gap mix: 0.5 metres, 0.5 seconds), true label unlabelled

| second | speed km/h | accel m/s2 | gap m | car ahead | lane offset m | speed term | accel² term | gap term | lane term |
|---|---|---|---|---|---|---|---|---|---|
| 601 | 27.2 | -0.15 | 11.8 | yes | unknown | 0.181 | 0.0010 | 0.407 | unknown |
| 602 | 27.2 | 0.00 | 12.5 | yes | unknown | 0.181 | 0.0000 | 0.380 | unknown |
| 603 | 27.2 | -0.05 | 12.8 | yes | unknown | 0.182 | 0.0001 | 0.372 | unknown |
| 604 | 26.8 | -0.19 | 12.7 | yes | unknown | 0.179 | 0.0015 | 0.372 | unknown |
| 605 | 25.8 | -0.27 | 13.0 | yes | unknown | 0.172 | 0.0030 | 0.353 | unknown |
| 606 | 24.8 | -0.30 | 13.0 | yes | unknown | 0.166 | 0.0037 | 0.342 | unknown |
| 607 | 23.6 | -0.37 | 13.0 | yes | unknown | 0.158 | 0.0054 | 0.333 | unknown |
| 608 | 22.2 | -0.43 | 13.2 | yes | unknown | 0.148 | 0.0074 | 0.312 | unknown |
| 609 | 20.5 | -0.53 | 12.8 | yes | unknown | 0.137 | 0.0112 | 0.309 | unknown |
| 610 | 18.4 | -0.79 | 10.8 | yes | unknown | 0.123 | 0.0252 | 0.350 | unknown |

| term | mean over the window | weight | points (weight x term / known weights x 100) |
|---|---|---|---|
| speed | 0.1626 | 0.25 | 4.5 |
| accel2 | 0.0058 | 0.35 | 0.2 |
| prox2 | 0.3530 | 0.30 | 11.8 |
| wave | unknown | 0.10 | unknown |

known weights add up to 0.90. score = 16.5, label **conservative** (cut offs 19.0 / 27.5). largest part: prox2 (11.8 of 16.5 points).
