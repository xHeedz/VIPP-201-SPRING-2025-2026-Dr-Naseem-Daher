# **F. Traffic flow modeling and microscopic simulation**

Three references on the physics of how heterogeneous traffic flows. The IDM and MOBIL papers are the foundation for the surrounding-vehicle behavior in the SUMO scenarios; Kesting & Treiber's textbook is the standard reference for shockwave dynamics, which is what the heterogeneous-driver experiment in week 8 measures.

**[20] **M. Treiber, A. Hennecke, and D. Helbing, "Congested traffic states in empirical observations and microscopic simulations," Physical Review E, vol. 62, no. 2, pp. 1805-1824, 2000.

***Used in this project: ****[seminal] The Intelligent Driver Model (IDM) paper. The basis for surrounding-vehicle longitudinal behavior in SUMO scenarios. The May 4 IDM/MOBIL configuration pass walked through every vType in every routes file to make sure all surrounding vehicles use this model consistently. Cited in the March Monthly Report and re-cited in the May Coda.*

**[21] **A. Kesting, M. Treiber, and D. Helbing, "General lane-changing model MOBIL for car-following models," Transportation Research Record, vol. 1999, no. 1, pp. 86-94, 2007.

***Used in this project: ****[seminal] The MOBIL lane-change model paper. Pairs with IDM for surrounding-vehicle behavior: IDM handles longitudinal car-following, MOBIL handles lane-change decisions. Both were configured explicitly on every vType in the May 4 fix. Cited in the May Coda.*

**[22] **A. Kesting and M. Treiber, Traffic Flow Dynamics: Data, Models and Simulation. Berlin: Springer, 2013.

***Used in this project: ****The standard textbook for traffic-flow dynamics. Used as the reference for shockwave-propagation expectations during the week-8 heterogeneous-driver experiment. The general finding from the textbook (aggressive drivers raise local accelerations and braking events for nearby vehicles, with magnitude depending on density) is consistent with the +6.8 AI bump we measured. Our specific magnitude is a new measurement on a new metric.*

VIPP 201A Annotated Bibliography  —  Hadi Al Shmaissani  —  Page  of
