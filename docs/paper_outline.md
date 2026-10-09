# paper outline: IEEE ITSC

6 pages (IEEE conference format, 2 extra pages allowed at a charge). every number below is in `data/` and in the research notebook pages of Oct 2026; nothing goes into the paper without its hand check.

## working title

- "When is a driver aggressive? A verified kinematic index with environment dependent proximity and a sequential RL labeler"
- shorter: "Sequential labeling of driver aggressiveness with reinforcement learning"

## claims (each with the evidence already in hand)

1. a closed form index that is hand checked end to end on real and simulated data (UAH, NGSIM US-101, Lankershim, pNEUMA, SUMO): paper number equals code number to 2 decimals. `docs/hand_checks/`, `tests/test_hand_checks.py`
2. proximity in metres reads dense traffic as aggressive (normal SUMO drivers +16.5 points from light traffic to a jam; NGSIM: congested traffic outscores UAH aggressive drivers); time headway removes the effect (-1.6) but loses on UAH; an environment dependent mix (half metres on highway and urban, 20% in weather) gives the best SUMO AUC on highway (0.869 against 0.750 metres, 0.789 headway) and halves the jam effect (+8.7). `scripts/fit_proximity_mix.py`
3. aggressiveness estimation as a sequential decision problem: the agent chooses when it has seen enough of a driver. on planted SUMO drivers with headway features, PPO reaches 0.736 balanced accuracy (3 seeds) after about 30 s, where the per environment index reaches 0.677 at 30 s and 0.741 at 60 s; it catches 83 to 88% of aggressive drivers. `scripts/train_labeling_rl_planted.py`
4. two training details that decide whether the sequential agent works at all: no discount (gamma 0.99 makes a label after 60 s worth 0.55) and an initial wait bias (without it, 44 to 72% of decisions come after one second). both shown with an ablation
5. honest negative result: on UAH (28 trips, trip level labels) the RL labeler does not beat the supervised per environment weights (0.792 vs 0.804 at 10 s)

## sections

1. introduction: why a driver score needs context (same 1 s headway, motorway vs jam); what is new (claims 2 to 4)
2. related work: driver behaviour scoring (bibliography A, H), traffic flow and density (F), RL for driving and for sequential classification / early classification of time series (B, C), context and retrieval (J). add 2 to 3 papers on early classification and optimal stopping (to find)
3. the index: four terms, normalisation, weights, cut offs fitted by Youden J (29 / 42); hand check table for one window
4. proximity: metres, time headway, mix per environment; table of AUC by alpha and environment; density effect figure
5. sequential labeling: MDP (observation, actions, reward, horizon), PPO settings, gamma and wait bias ablation
6. data: UAH (labelled, real), planted SUMO drivers (three types, 7 settings, 10 seeds, tau calibrated to NGSIM headway), NGSIM / pNEUMA (unlabelled, density); highD once scored
7. results: table of RL vs fixed window baselines (balanced accuracy, per class accuracy, seconds, reward); one traced decision (normal driver above the 42 cut off, labelled normal after 21 s); UAH negative result
8. limitations: simulated labels, trip level UAH labels, US-101 speed distribution not matched (best speed KS 0.37), no real dense traffic with labels yet
9. conclusion

## figures (max about 6)

- [ ] index pipeline with the hand check numbers of one window
- [ ] density effect: normal driver score vs traffic density, metres / headway / mix
- [ ] AUC vs alpha per environment
- [ ] MDP diagram: observe, wait or label
- [ ] RL vs fixed window: balanced accuracy against seconds to decision (one point per method)
- [ ] traced decision: probabilities of wait and normal over time for the normal driver

## missing before submission

- [ ] highD: score in metres, headway and mix; check the density effect on German motorway data
- [ ] DriveDNA (full): real car following with radar gap, more drivers
- [ ] RL labeler on real labelled data beyond UAH (none with per driver labels in dense traffic yet)
- [ ] gamma and wait bias ablation over 3 seeds each
- [ ] Dr. Daher's approval of the framing and the author list
- [ ] related work on early classification / optimal stopping
