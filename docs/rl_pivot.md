# rl pivot: turning the aggressiveness index into a real rl problem

Handoff for Claude Code. Read `CLAUDE.md` first (working style, writing style, machine setup), then this file. The working style rules in `CLAUDE.md` apply here too: explain every change before making it (file, line, current behavior, new behavior), point to file and line for "what calls what", run commands and report the numbers, and ask before git commit or push.

## why this exists

Hadi's part of the project computes the Aggressiveness Index (AI). A code audit on 5 Oct 2026 showed that nothing on the index side is reinforcement learning, even though the spring reports describe it as RL. Hadi wants the index side to become a real RL problem, not supervised learning with an RL label.

## audit findings (verified in code, 5 Oct 2026)

1. `model/dynamic_weight_agent.py`: purely supervised.
   - The model is 3 sets of 4 softmax weights (highway, urban, weather) plus one threshold per environment and a shared `log_tau`. It is not the 9 to 64 to 64 to 4 MLP described in the final report.
   - Spring: weights fit by MSE against ground truth scores (`experiments/weight_agent/train_strict.py`).
   - Fall: `fit_labels` (around line 63) fits weights and thresholds with balanced binary cross entropy on UAH normal/aggressive labels.
   - No KL anchor term, no reward, no REINFORCE, no alpha annealing anywhere in it.
2. `agent/driver_q_learner.py`: called Q-learning, but is supervised classification.
   - State = AI score bucket (`AI_Score // 10`), action = category label, reward is always 15, and only the correct label cell is updated (line 32). The table converges to counting the most frequent label per bucket. No environment, no exploration, no real reward signal.
   - It is also circular: the label is itself a threshold on the same AI score used as the state.
   - The count based curiosity bonus (beta / sqrt(N(s)+1)) described in the reports and bibliography is not in this file.
3. Real RL exists only on the driving side: `env/scenarios/*.py`, `scripts/sumo_highway_agent.py`, `scripts/sumo_urban_agent.py`, `scripts/highway_env/*.py`. These are REINFORCE with sampled actions, `log_prob`, discounted returns and an entropy bonus. That policy drives the ego car; the AI is only one input to its state. This is not Hadi's part.
4. Consequence: the final report's description of the index model (MLP, KL plus RL loss, curiosity bonus) does not match the code. Do not repeat those claims in any new document. Describe only what the code does.

## the core design constraint

The index must not drive the car. Hadi's part only estimates aggressiveness. So the RL formulation has to make "estimating aggressiveness" itself a sequential decision problem with a real reward, not wrap a supervised loss in RL vocabulary.

## candidate formulations (present these to Hadi and let him choose before writing code)

### option a: sequential labeling (recommended first step)

- Episode = one driver observed second by second (UAH trip windows, planted SUMO drivers, later NGSIM).
- Observation at time t: running index features (n_speed^2, n_accel, n_prox^2, n_wave) as rolling stats, current AI from `model/aggressiveness_model.py`, environment one hot, seconds observed so far.
- Actions: `wait`, `label_conservative`, `label_normal`, `label_aggressive`. A label action ends the episode.
- Reward: +1 correct label, -1 wrong label (asymmetric costs allowed, e.g. a missed aggressive driver costs more), small cost per `wait` step, forced decision at a max horizon.
- Why it is real RL: actions change future observations (waiting reveals more driving), reward is delayed, and the agent must trade speed of detection against accuracy. Supervised learning cannot learn when to commit.
- Labels already exist: UAH normal/aggressive trips (`datasets/uah.py`), planted SUMO drivers with known types (`scripts/planted_drivers.py`, `data/planted_runs.csv`).
- Algorithm: DQN or PPO via stable-baselines3 on a gymnasium env. Pin versions that support Python 3.9.6 (the Mac venv).
- Baselines to beat: the current fixed threshold index and the DynamicWeightAgent at a fixed window length. Report accuracy and mean seconds to decision.

### option b: inverse RL on real trajectories (strongest paper angle)

- Treat each driver as an agent maximizing a reward over features like speed, time headway, acceleration and lane changes (`model/aggressiveness_model.py` `headway_features` is a starting point).
- Recover per driver reward weights with maximum entropy IRL from NGSIM (and highD/exiD once access arrives). Aggressiveness = position in the learned reward weight space, replacing the hand picked weights (0.5, 0.2, 0.8, 0.4).
- Heavier: needs a discretized or simulated driving MDP (SUMO or highway-env) for the inner RL loop.
- Good fit for the IEEE ITSC / T-ITS paper and answers "where do the weights come from".

### option c: closed loop weight tuning (not recommended)

- Index weights as actions, reward from how safely a downstream planner drives when using the index.
- Rejected by default: it ties the index to the driving policy, which is outside Hadi's part.

Suggested order: build option a first (fast, uses existing labels), keep option b as the paper direction.

## constraints from Dr. Daher (early Oct 2026 feedback)

- Hand check every number before anything new is presented. The RL work must not hide unverified index values.
- Explain the very high AI values first (see the 25 Sep NGSIM finding in `CLAUDE.md`).
- Train on more and better datasets.
- RAG for context (`model/rag.py`) stays on the plan; an RL observation can later include RAG context features.
- Confirm the RL direction with Dr. Daher at the next meeting before investing heavily.

## first session plan

[ ] walk Hadi through the audit findings above in the code (file and line), so he can confirm them himself
[ ] present options a, b, c; Hadi picks
[ ] if option a: write `env/labeling_env.py` (gymnasium env), unit tests in `tests/test_labeling_env.py`, explain each method before writing it
[ ] train a first DQN or PPO agent on UAH windows with leave one driver out, same split as `scripts/train_uah.py`
[ ] compare against the fixed threshold index and the DynamicWeightAgent; report accuracy and seconds to decision honestly, including if RL does worse
[ ] decide what to do with `agent/driver_q_learner.py`: rename it to describe what it does, or replace it with the new agent; do not keep calling it Q-learning
[ ] add a short note to `docs/next_session.md` with the decision and results
