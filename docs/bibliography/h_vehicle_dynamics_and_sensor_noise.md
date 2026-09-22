# **H. Vehicle dynamics and sensor noise models**

Five references covering vehicle dynamics (Rajamani, Astrom & Murray) and the sensor-noise modeling literature behind the May 2-9 integration push (Winner, Pomerleau, Thrun et al.). The May 2-9 work introduces a Gaussian sensor-noise model with σ_vel = 0.5 m/s, σ_gap = 0.8 m, σ_acc = 0.3 m/s² — these magnitudes are calibrated against published radar/lidar specifications.

**[28] **R. Rajamani, Vehicle Dynamics and Control, 2nd ed. Springer Science & Business Media, 2011.

***Used in this project: ****[seminal] The standard textbook for vehicle dynamics. Cited as the reference for the friction-circle and slip-ratio physics underlying the friction-aware regime in week 11. The slip-ratio derivation from kinematic observables (slip ≈ |v_actual − v_intended| / v_actual) is consistent with this textbook's treatment.*

**[29] **K. J. Aström and R. M. Murray, Feedback Systems: An Introduction for Scientists and Engineers, 2nd ed. Princeton University Press, 2021.

***Used in this project: ****Background reference for control-system fundamentals. Cited in the Milestone-1 report. Not used directly in term-2 work but kept in the bibliography because the closed-form AI score is fundamentally a feedback signal that downstream planners would close a control loop around.*

**[30] **S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics. Cambridge, MA: MIT Press, 2005.

***Used in this project: ****[seminal] The standard reference for probabilistic state estimation. Used as the conceptual basis for the SumoAgentAssessor's noise model in the May 2-9 work — the assessor adds Gaussian noise to telemetry inputs before running the formula, which is exactly the perception-model framing this textbook formalizes.*

**[31] **H. Winner, S. Hakuli, F. Lotz, and C. Singer, Eds., Handbook of Driver Assistance Systems: Basic Information, Components and Systems for Active Safety and Comfort. Springer, 2016.

***Used in this project: ****The reference for automotive sensor characteristics. The σ values in the May 2-9 noise model (σ_vel = 0.5 m/s for radar Doppler, σ_gap = 0.8 m for lidar/radar range) are calibrated against the chapters on automotive radar and lidar measurement noise in this handbook. Cited in the May Coda.*

**[32] **D. Pomerleau, "Visibility estimation from a moving vehicle using the RALPH vision system," in IEEE Conf. Intelligent Transportation Systems, 1997.

***Used in this project: ****Older but still useful reference for sensor-noise characterization in moving-vehicle perception. Cited in the May Coda alongside Winner et al. for the sensor-noise framing. Honestly more historical than load-bearing.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
