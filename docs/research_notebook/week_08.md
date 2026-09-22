# **Week 8: Tuesday Mar 10 – Monday Mar 16, 2026**

**Theme — ***Heterogeneous SUMO drivers + the shockwave hypothesis. The big result of the term lands.*

## **Goals for the week**

Run the heterogeneous-driver experiment. 5% aggressive vehicles in a SUMO highway scenario, 95% normal. 5 minutes of sim time. Compute per-vehicle mean AI score. Two things to confirm: (a) aggressive vehicles score higher than normals (should be obvious — they're parameterized to be more aggressive), and (b) the *normal* vehicles spatially adjacent to aggressive ones score higher than the reference normals from an aggressive-free run. The second one is the shockwave hypothesis from the midterm's future-work section.

## **Daily entries**

**Tue Mar 10***  —  Heterogeneous scenario authoring.*

Defined two vType blocks in the SUMO routes file. Normal: minGap=2.5, accel=2.5, decel=4.5, impatience=0.0, sigma=0.5. Aggressive: minGap=0.5, accel=4.0, decel=4.5, impatience=1.0, sigma=0.7. The probability flag in the demand schedule controls the aggressive fraction — set to 0.05 to get 5% aggressive vehicles drawn from the joint demand distribution. Verified visually in the SUMO GUI: aggressive vehicles tailgate noticeably and lane-change more often. Looks right.

Hit a snag with the demand file format — `vTypeDistribution` is the right element to use, not what I had at first. Fixed and confirmed by parsing the resulting tripinfo XML and counting type assignments. Got 4.8% aggressive (close enough to 5% given the Poisson spawn process). 5-min run produces about 600 unique vehicles total.

**Wed Mar 11***  —  Per-vehicle AI computation.*

Wrote scripts/sumo_per_vehicle_ai.py. Iterates through the trajectory log, groups telemetry samples by vehicle ID, computes mean AI score per vehicle. Cross-checked against vehicle type. Aggressive vehicles: mean AI = 71.4 (sd 8.3). Normal vehicles in this run: mean AI = 33.1 (sd 6.7). Big separation, as expected.

Now the interesting part. Need a control: aggressive-free reference run. Re-ran with 0% aggressive (everyone normal) for the same 5 minutes. Reference normal mean AI = 26.3. So normals in the heterogeneous run score 33.1 vs. 26.3 in the homogeneous run — a 6.8 AI-point bump. That's the shockwave.

**Thu Mar 12***  —  Spatial decomposition of the shockwave.*

Decomposed the 6.8-point bump spatially. For each normal vehicle, computed minimum distance to any aggressive vehicle averaged over the trajectory; binned normals into "close" (avg distance < 30m at any point), "medium" (30-100m), and "far" (>100m). Mean AI scores: close = 38.7 (+12.4 over reference), medium = 32.9 (+6.6), far = 27.8 (+1.5). So the shockwave is real and it has a clean spatial decay. The far group is barely elevated above reference, which means we can attribute most of the population-level bump to vehicles that actually came into proximity with an aggressive driver.

Wrote it up with a histogram per bin overlaid against the reference normal distribution. The close-distance histogram has a visibly heavier right tail. This is the cleanest empirical thing the project has produced so far.

**Fri Mar 13***  —  Lit check + statistical test.*

Looked up the traffic-flow literature on shockwave propagation — Kesting & Treiber 2013 textbook, Treiber et al. 2000 (the IDM paper), and a couple more recent ones on car-following heterogeneity. The general finding is that aggressive drivers in heterogeneous fleets do raise local accelerations and braking events for nearby vehicles, but the magnitude varies a lot with density. Our setup is moderate density. Our 6.8-point population bump is consistent with the general direction the literature predicts — we're not contradicting anything established, but the magnitude on our specific AI metric is a new measurement.

Did the statistical test — two-sample t-test on the close-bin normal AI scores vs. reference normal AI scores. t = 9.8, p < 1e-15. Effect is solid. The cohen's d is also large (~1.5), so this isn't just statistical significance from a big N — the effect size is real.

**Sat Mar 14***  —  Off.*

Off.

**Sun Mar 15***  —  Wrote up week 8 for the sync.*

Spent 2 hours writing up the heterogeneous-driver experiment for the Monday sync. Three figures: (1) per-vehicle AI distribution colored by vehicle type, showing the aggressive cluster cleanly separated from normal; (2) the spatial-bin histogram showing the shockwave with distance-decay; (3) trajectory-overlay plot showing one specific aggressive vehicle and the elevated AI scores of normals in its 30m radius over time. The third figure is the most evocative — you can see the AI score of a tailing normal vehicle jump within ~2 seconds of the aggressive vehicle entering its proximity zone.

**Mon Mar 16***  —  Sync.*

Sync. Walked Daher through the three figures. He paused on the third one for a long time. His comment: "This is publishable." He wants me to do two follow-ups for week 9: (a) two-hop propagation — does a normal that came into proximity with a normal that came into proximity with an aggressive (i.e. two hops removed from the source of aggression) also show elevated AI? Hypothesis: yes, but with a smaller magnitude. (b) Sweep aggressive fraction from 0% to 20% in 5% steps, plot population-level mean AI as a function of aggressive fraction. Hypothesis: monotone increasing, possibly nonlinear above some critical fraction.

Also flagged that I should think about whether this finding suggests a control intervention — e.g., if a fleet of partially-autonomous vehicles could detect aggressive neighbors via the AI score and adjust to dampen the shockwave. He's not asking for it this week but he wants me to start thinking about it for the multi-objective reward shaping later in the term.

## **Reflection**

This is the headline week. The shockwave hypothesis got a clean, statistically significant confirmation with a clear spatial decay. "This is publishable" was the line from Daher and that's not something he says lightly. The fact that I can attribute the population-level effect to specific spatial-proximity events is what makes the result feel solid rather than coincidental. Going to ride this momentum into the dose-response sweep next week.

## **Plan for next week**

Two-hop propagation experiment + aggressive-fraction sweep (0%, 5%, 10%, 15%, 20%) over 5-minute scenarios each. Plot the dose-response curve. If it looks nonlinear, check whether a critical fraction exists where the system tips into broader propagation.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
