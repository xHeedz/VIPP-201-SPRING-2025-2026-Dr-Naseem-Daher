# **G. Simulation tools and middleware**

Five references covering the simulation and middleware stack: highway-env (Phase-1 simulator), SUMO (term-2 main simulator), CARLA (descoped stretch goal), and ROS / ROS2 (the publishing layer). All are tooling references rather than methodological ones.

**[23] **E. Leurent. (2018). An environment for autonomous driving decision-making (highway-env). [Online]. Available: https://github.com/eleurent/highway-env

***Used in this project: ****The Phase-1 simulator. Used throughout the term for controlled scenarios where lightweight 2D kinematics are sufficient (lane-change tests, agent verification, the controlled-flow midterm dataset). The transition to SUMO in week 7 was for population-scale experiments that highway-env's ~15-vehicle limit can't support.*

**[24] **P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y.-P. Flötteröd, R. Hilbrich, L. Lücken, J. Rummel, P. Wagner, and E. Wießner, "Microscopic traffic simulation using SUMO," in 21st IEEE International Conference on Intelligent Transportation Systems, 2018, pp. 2575-2582.

***Used in this project: ****[seminal] The SUMO simulator paper. SUMO is the term-2 main simulator and the basis for the heterogeneous-driver experiment, the dose-response sweep, the Social Latency mitigation, the friction-aware regime, and the May 2-9 noise-validation scenarios. TraCI (the Python client) is the API used throughout.*

**[25] **A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, "CARLA: An open urban driving simulator," in Conference on Robot Learning, 2017, pp. 1-16.

***Used in this project: ****Cited as the descoped stretch-goal simulator. The April Monthly Report and the Final Report's future-work section both flag CARLA integration as the most important future-work item — it would unlock the Advanced Urban Predictive Parameters (Occlusion-Aware Speed Factor, PET, VRU Potential Field) defined in the midterm but not yet validated. The week-9 descoping decision is recorded as a deliberate trade-off rather than a failure.*

**[26] **S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "The Robot Operating System 2 (ROS 2): Design, architecture, and uses in the wild," Science Robotics, vol. 7, no. 66, 2022.

***Used in this project: ****[seminal] The ROS 2 architecture paper. Used as the reference for the rclpy + colcon + custom message types pattern implemented in week 12's ROS 2 wrapper. The mature design of ROS 2 is what made the wrapper week one of the cleanest weeks of the term.*

**[27] **M. Quigley, K. Conley, B. Gerkey, J. Faust, T. Foote, J. Leibs, R. Wheeler, and A. Y. Ng, "ROS: An open-source robot operating system," in ICRA Workshop on Open Source Software, vol. 3, no. 3.2, 2009.

***Used in this project: ****The ROS 1 origins paper. Cited in the Milestone-1 report for completeness of the middleware lineage; the actual implementation uses ROS 2.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
