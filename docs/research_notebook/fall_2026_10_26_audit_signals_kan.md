# mon 26 oct 2026: label audit, new signals, KAN, training length, FSM, German transfer

requests (Dr. Daher, meeting of 26 oct): check by hand where the model and the labels disagree (random samples, time series); FSM switching between scenario experts, each trained and tested on its own, compare with fusion; how to move the model to German data (one hot, relative speeds); more complex reward (TTC and more); training length; physics informed signals; where the RAG is. Dr. Kurdahi: KAN (learned activation functions, grid size and degree), interpretability, index as components, intervention. Choices by Hadi: experts + FSM; German = simulated autobahn + highD proxy labels; risk weighted reward; KAN this week.

## 1. label audit (UAH)

`scripts/label_audit_uah.py` -> `../26-10-2026/label_audit/` (figures, 10 s clips, index.html, review.csv), `data/label_audit_summary.csv`. formula (reference, aggressive from 27.5) against the trip label, DriveSafe (SEMANTIC_ONLINE column 14, aggressive from 0.5) as third opinion. video time = data time + (data start - video start), as in the UAH reader tool.

| group | windows | DriveSafe says aggressive | DriveSafe sides with the formula |
|---|---|---|---|
| label aggressive, formula calm | 325 (21% of aggressive) | 28% | 72% |
| label normal, formula aggressive | 697 (27% of normal) | 0% | 0% |
| agree aggressive | 1,243 | 52% | |
| agree normal | 1,903 | 0% | |

DriveSafe never flags a second of a normal trip, so it cannot judge the second group. 22 windows sampled (seed 0). examples: 02 (label aggressive, formula 23.8): gap closes 50 to 10 m at the end of the window, UAH TTC about 1 s: label probably right, formula late. 10 (label normal, formula 31.7): 100 km/h in a 120 zone behind trucks: label right, formula wrong (raw speed). manual verdicts pending (Hadi, review.csv).

## 2. training length

`scripts/learning_curve.py`, `data/learning_curve_{ppo_small6_wait5,dqn_full}.csv`. 1 step = 1 s of one driver and one action; 1M steps about 30,000 drivers, 2.6 min on the laptop (pre recorded tables, 64 x 64 network). PPO 0.72 at 0.5M, between 0.71 and 0.74 up to 10M (0.713 at 10M, 27.7 min). DQN explores until 3M (0.37 to 0.50), then 0.70 to 0.76. longer training does not raise accuracy.

## 3. new signals, extended formula

`datasets/signals.py` (tests `tests/test_signals.py`, hand values incl. IDM -3.82 m/s2), clean tables now carry speed_limit_kmh (UAH OpenStreetMap, SUMO edges, levelX recorded, NGSIM 65 / 35 mph and pNEUMA 50 km/h assumed) and traffic_kmh (mean speed of the other vehicles at the same second). terms: TTC (1 - TTC / 4)^2, speed vs limit / 0.3, speed vs traffic / 0.3, |IDM residual| / 3 (a_max 1.5, b 2.0, T 1.5 s, s0 2 m, v0 = limit).

single term AUC (`data/extended_single_terms.csv`), UAH / SUMO A vs N: speed 0.690 / 0.547, accel^2 0.731 / 0.587, gap 0.792 / 0.767, lane 0.542 / 0.558, TTC 0.513 / 0.494, speed vs limit 0.676 / 0.564, speed vs traffic none / 0.574, IDM 0.784 / 0.885. TTC averaged over 10 s carries nothing (closing moments are short); IDM partly by construction on SUMO (IDM drivers).

`scripts/fit_extended_weights.py` (8 terms, 0.1 grid, 19,448 vectors): weights 0.2 / 0.2 / 0.2 / 0.1 / 0.1 / 0.1 / 0.0 / 0.1. UAH LODO 0.860 (reference 0.842; D1 0.840 vs 0.761), SUMO A vs N 0.914 (0.888), N vs C 0.787 (0.791).

## 4. KAN

`model/kan.py` (additive KAN, cubic B-splines, grid 5, SiLU base, masked unknown inputs; tests `tests/test_kan.py`), `scripts/kan_formula.py`, unsquared window inputs (`datasets/windows.py` raw_inputs). AUC: UAH LODO 0.894 (reference 0.842, extended 0.860; better on all six drivers), SUMO A vs N 0.917 (0.858 / 0.889). learned shapes (`data/kan_formula.json`, `results/figures/kan_curves.png`): acceleration linear on UAH (R^2 1.00), sqrt on SUMO (1.00), not squared; speed cubic on UAH (0.99), square on SUMO (1.00); gap square / cube; lane linear; speed vs limit sqrt / linear; headway U shaped (minimum near 1.5 s); TTC curve on very few points, not interpretable.

## 5. agents: risk reward, extended state, FSM experts

`env/labeling_ref.py`: extended sequences (IDM term, speed vs limit and traffic, danger = max(TTC danger, (1 - headway / 1 s)^2)), state EXT_FEATURES, risk reward (wait x (1 + 5 danger), missed aggressive -(1 + 2 max danger)), country one hot; tests in `tests/test_labeling_ref.py`. `scripts/fsm_vs_joint.py` (`data/fsm_vs_joint*.csv`), SUMO test seeds 7 to 9:

| agent | all | highway | urban | weather | dangerous aggressive caught / seconds (58 drivers) |
|---|---|---|---|---|---|
| previous best (8 inputs) | 0.722 | | | | |
| joint, extended state, plain reward | 0.795 | 0.816 | 0.812 | 0.688 | 0.879 / 22.3 |
| joint, risk reward | 0.797 | 0.816 | 0.831 | 0.691 | 0.879 / 19.3 |
| joint, risk, 3M steps | 0.798 | 0.811 | 0.821 | 0.719 | 0.897 / 14.4 |
| 3 experts + FSM | 0.801 | 0.801 | 0.875 | 0.751 | 0.897 / 15.9 |

new information: +7 points; reward and length: +0.2. median score at decision per label (experts): highway 17.2 / 21.3 / 25.5, urban 11.8 / 16.2 / 22.8, weather 14.4 / 19.0 / 21.8 (conservative / normal / aggressive): same ordering, thresholds shifted per road type, so fusion = shared agent + road type threshold.

## 6. German transfer

autobahn scenario (`scripts/planted_drivers.py`, `data/sumo_planted_de/`, clean source planted_de): 3 lanes, 2 km at 120 km/h, 2 km without limit (desiredMaxSpeed 140 km/h x speedFactor), lcKeepRight 2, lcOvertakeRight 0.5 aggressive / 0.05 normal / 0, 15% trucks at 80 km/h, 3000 veh/h, 10 seeds. medians on the unlimited section: aggressive 128, normal 118, conservative 110, trucks 79 km/h.

`scripts/german_transfer.py` (`data/german_transfer*.csv`), autobahn test seeds 7 to 9 (1,545 vehicles): A other roads, old inputs 0.726; B other roads + relative speeds / IDM / danger 0.808; C + autobahn seeds 0 to 6 + country one hot 0.738 (decides at 18 s); D autobahn only 0.722 (17 s). formula on autobahn windows: reference 0.855, extended 0.839. relative features transfer; C and D single runs on a small German training set (3,824 vs 22,811), to repeat with more seeds.

highD proxy labels (81,310 windows, 25.6% proxy aggressive): all terms reference 0.975, extended 0.949 (circular); without gap and speed vs limit: reference 0.616, extended 0.833 (IDM single 0.933, partly overlapping with close following).

## 7. RAG

`model/rag.py` used only by `scripts/rag_ablation.py` (4 oct): not part of the formula, the supervised model or the agents. plan: context percentile among retrieved normal windows (road type, traffic state, speed limit, country) as an agent state input; knowledge files only for explanations.

## 8. deck

`scripts/make_presentation_figures.py` (results/figures/pres_*.png), `scripts/build_long_deck.py` -> `../26-10-2026/Progress Presentation.pptx` (18 slides, figures / equations / videos, explanation in the notes).

## status

- [x] label audit with figures and video clips
- [x] learning curves to 10M steps
- [x] TTC, relative speeds, IDM residual; extended formula
- [x] KAN learns the formula
- [x] risk reward, extended state, FSM experts vs joint
- [x] German autobahn scenario, transfer, highD proxy
- [ ] manual verdicts in review.csv (Hadi)
- [ ] Charbel's slide style (file from Hadi)
- [ ] repeat C (German data + one hot) with 3 seeds
- [ ] RAG context percentile in the agent state
- [ ] Dr. Daher: acceleration straight and speed curved (KAN result)
