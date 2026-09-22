# **Week 5: Tuesday Feb 17 – Monday Feb 23, 2026**

**Theme — ***Diagnostics: mixture-weight panel, Bellman-error tracking, state-visitation heatmap. The diagnostics break the model.*

## **Goals for the week**

Two diagnostics to write this week. (1) The mixture-weight panel Daher asked for — plot the gate's 3-way mixture over time during a regime-changing episode, confirm the soft handoff is smooth and not noisy. (2) Bellman-error tracking on the Q-learner — for every (s, a) update, log |target - Q(s,a)| and plot its distribution over training. The point is to identify states where the agent is consistently confidently wrong, which is the textbook "memorizing not learning" signature Daher flagged. Bonus: a state-visitation heatmap across the 4D normalized telemetry space, to see whether the agent is just specializing on a thin slice of the state distribution.

## **Daily entries**

**Tue Feb 17***  —  Mixture-weight panel.*

Wrote eval/plot_mixture_weights.py. Three subplots: (top) the input context vector channels over time, (middle) the gate's three mixture weights over time, (bottom) the four predicted AI weights over time. All three share an x-axis. Ran it on a 90-second synthetic episode that cycles highway → urban → weather → highway. The mixture-weight subplot is genuinely satisfying to look at — clean sigmoid handoffs centered exactly at the regime-flip events in the context stream, with the highway and urban mixtures crossing over cleanly while weather sits near zero, then weather rises smoothly during its window. Saved as eval/mix_weights_3regime.png.

**Wed Feb 18***  —  Bellman error: instrumentation.*

Pulled up the old DriverQLearner and added Bellman-error logging. Ramadan started today so I shifted my schedule a bit — got to lab around 11am after a late suhoor. For each Q-table update we already compute target = r + γ·max_a' Q(s', a') and the TD error δ = target - Q(s, a). I'm now appending δ and (s, a) to a CSV every step, then plotting the distribution of |δ| as a histogram and as a function of training episode.

Re-ran the Q-learning experiment from the midterm. Bellman error histogram is heavily right-skewed — 95% of |δ| < 0.05, but a long tail out to 0.6. The high-|δ| samples are concentrated in the (low n_speed, high n_prox) corner of state space. Translation: the agent is most confused in dense urban-like states, which is exactly where the Phase-1 dataset is sparsest. So the agent isn't memorizing well-explored states either — it's *confidently wrong* in under-explored states. That's worse than the midterm story made out.

**Thu Feb 19***  —  State-visitation heatmap.*

Wrote the state-visitation heatmap. Four 2D marginals (each pair of normalized telemetry channels), each a 32×32 grid, each cell counting how many times the agent visited that bin during 1000 training episodes. The heatmap is genuinely uncomfortable to look at: the (n_speed, n_prox) marginal has ~70% of all visitation concentrated in the (n_speed > 0.7, n_prox < 0.3) corner — fast and unobstructed, the easy regime. Almost no visitation in the high-n_prox column where the Bellman error is also worst. The agent has learned to drive in such a way that it stays in the high-speed low-density region of state space, where its predictions are good, and avoids exactly the urban-like states where they aren't.

This is the conservative-bias problem. The Q-learner has implicitly learned a policy that minimizes the chance of finding out it's wrong. That's a textbook exploitation-vs-exploration failure and it's exactly what Daher was pointing at in January. The accuracy curve looks great because the agent has effectively chosen its own evaluation distribution.

Honestly, I sat with this for a while. It's a real result. I'm going to write it up properly for the Monday sync and not try to dress it up.

**Fri Feb 20***  —  Wrote up the diagnostic.*

Wrote a 2-page diagnostic memo with the three figures (mixture-weight panel, Bellman-error histogram, state-visitation heatmap) and a paragraph each explaining what they show. The honest version of the story: the gate-head agent (with supervised KL loss) is actually fine — its predictions track context cleanly. The Q-learner isn't fine — it's exploiting an exploration loophole that the midterm's metrics didn't catch.

The fix has to involve some kind of intrinsic motivation that pulls the Q-learner into under-visited states. That's the curiosity-driven exploration thing I had in the lit triage notes. Time to do something about it.

**Sat Feb 21***  —  Coffee + paper triage.*

Re-read Bellemare et al. 2016 "Unifying count-based exploration and intrinsic motivation" and Pathak et al. 2017 "Curiosity-driven exploration by self-supervised prediction." For our setting, count-based is the right pick — small discrete state space (after we discretize the 4D normalized telemetry), already tabular, doesn't need a learned predictor. The Pathak paper is for high-dim observations like pixels; overkill here. Wrote up a 1-page proposal: discretize each of the 4 telemetry channels into 10 bins → 10⁴ = 10000 cells → maintain a visitation count N(s) per cell → bonus reward = β / sqrt(N(s) + 1) added to the environment reward. β is a hyperparameter, will need tuning.

**Sun Feb 22***  —  Off.*

Off.

**Mon Feb 23***  —  Sync — and a real conversation about what the term is now.*

Sync was longer than usual — about 35 minutes. Walked Daher through the diagnostic memo. He read both the Bellman-error histogram and the state-visitation heatmap carefully and was quiet for a minute. His response, recorded as faithfully as I can: "This is what I was telling you in January. You found it. Now fix it." Agreed scope for week 6: implement the count-based curiosity bonus on the Q-learner, sweep β to find a value that pulls the visitation distribution closer to uniform without destabilizing training, re-run the same diagnostic and confirm Bellman errors are reduced in the high-prox corner.

*"**This is what I was telling you in January. You found it. Now fix it.**"*

## **Reflection**

This is the most important week of the term so far. Not because I built a lot — I didn't, the diagnostics are maybe 200 lines of code total — but because the diagnostics expose a real failure of the midterm's central claim. The Q-learner's accuracy was a function of where it chose to drive, not of how well it actually predicts. That's a hard thing to write down clearly but it's exactly the thing Daher kept asking me to look for. Curiosity bonus next.

## **Plan for next week**

Implement count-based curiosity. Discretize the 4D telemetry into a 10⁴ grid, maintain visitation counts, add β/sqrt(N+1) bonus to the reward. Sweep β ∈ {0.01, 0.05, 0.1, 0.5, 1.0}. Re-run training and re-plot the same three diagnostic figures. Pass: visitation distribution is more uniform across the (n_speed, n_prox) marginal; Bellman error in the high-prox column drops by at least 30%.

VIPP 201A Research Notebook  —  Hadi Al Shmaissani  —  Page  of
