# sun 4 oct 2026: part 4, retrieval for context (RAG)

idea: before scoring a window, retrieve normal windows recorded in the same situation and score the window relative to them. the same 1.0 s headway is aggressive at 120 km/h on an empty motorway and ordinary in a jam.

design diagram: `results/figures/rag_design.png`.

## what was built

- `model/rag.py`
  - `Context` (road type, speed limit, traffic state, manoeuvre, weather, region, vehicle class, density)
  - `WindowIndex`: reference windows with their raw index score and context; only NORMAL reference windows are used. match levels, most specific first: road + traffic + manoeuvre + weather, then road + traffic, then road, then all; inside a match the k = 200 nearest by speed limit (and density when both sides have it); at least 50 windows per level
  - context score = percentile of the window's raw score among the k neighbours; flag at 95
  - neighbours are chosen by context, not by driving features: neighbours in feature space drive like the window itself, so its percentile among them would sit near 50 for everybody
  - numpy distances instead of FAISS: exact and fast enough at this size (a few thousand to about 600k windows); FAISS can replace `query` without changing anything else
- `datasets/contexts.py`: `uah_context` (OSM speed limit, lane change events), `sumo_context`, `ngsim_context` (density to traffic state), `pneuma_context`
- `knowledge/`: one markdown file per region (spain, united_states, greece, lebanon, germany, simulation) with following distance guidance and default limits, each with sources. used only by `explain()`, never in the score
- `explain()`: context score, matched level and number of neighbours, largest term, time headway against the local guidance, naming the knowledge file. template text, no LLM (an LLM cannot sit in the 10 Hz ROS2 loop). examples: `docs/hand_checks/rag_explanations.md`
- tests: `tests/test_rag.py`

## ablation (`scripts/rag_ablation.py`)

UAH, leave one driver out (reference: normal windows of the other five drivers), AUC normal vs aggressive (`data/rag_ablation_uah.csv`):

| driver | no context | 3 environments (agent) | RAG | RAG, headway | road type only |
|---|---|---|---|---|---|
| D1 | 0.606 | 0.585 | 0.586 | 0.480 | 0.585 |
| D2 | 0.923 | 0.846 | 0.919 | 0.899 | 0.897 |
| D3 | 0.832 | 0.724 | 0.798 | 0.769 | 0.770 |
| D4 | 0.880 | 0.755 | 0.858 | 0.828 | 0.833 |
| D5 | 0.884 | 0.828 | 0.840 | 0.814 | 0.866 |
| D6 | 0.852 | 0.827 | 0.831 | 0.819 | 0.853 |
| pooled | 0.840 | 0.740 | 0.812 | 0.779 | 0.802 |

SUMO planted drivers (calibrated tau, see `fall_2026_10_04_open_items.md`), split by seed (reference seeds 0 to 4, test 5 to 9 and the reverse), AUC aggressive vs normal (`data/rag_ablation_sumo.csv`):

| | no context | no context, headway | 3 environments (agent) | RAG | RAG, headway |
|---|---|---|---|---|---|
| pooled over all scenarios | 0.759 | 0.802 | 0.712 | 0.790 | 0.806 |
| highway low | 0.885 | 0.843 | 0.883 | 0.887 | 0.846 |
| highway medium | 0.864 | 0.830 | 0.812 | 0.859 | 0.829 |
| jam | 0.743 | 0.759 | 0.636 | 0.758 | 0.754 |
| merge | 0.781 | 0.839 | 0.697 | 0.758 | 0.837 |
| roundabout | 0.924 | 0.975 | 0.884 | 0.925 | 0.968 |
| urban | 0.868 | 0.904 | 0.857 | 0.865 | 0.901 |
| weather | 0.805 | 0.824 | 0.833 | 0.801 | 0.824 |

(with the uncalibrated driver types, `data/rag_ablation_sumo_tau1.csv`: pooled 0.751 / 0.830 / 0.730 / 0.778 / 0.849.)

reading:
- inside one situation a percentile is a monotone rescaling of the raw score, so RAG cannot change the AUC much there (per scenario rows). it helps only where situations are mixed: pooled SUMO +0.031 (metres), +0.004 (headway).
- on UAH, RAG loses 0.028: UAH has two road types, no traffic state and free flow only, so there is little context to use, and many aggressive windows sit at the 100th percentile together (ties lose ranking information).
- measurement first: headway alone adds +0.043 on pooled SUMO; RAG adds +0.031 to the metres score and almost nothing on top of headway (both remove the same density effect). best: headway + RAG, 0.806.
- the per environment agent is the weakest everywhere except weather: three weight sets learned from trip labels do not carry the situation.

## NGSIM: what the reference set does (no labels, `data/rag_ngsim_flags.csv`, `data/rag_ngsim_flags_headway.csv`)

share of NGSIM windows flagged (context score >= 95):

| site | RAG metres, ref UAH | RAG metres, ref UAH + calibrated SUMO | RAG headway, ref UAH | RAG headway, ref UAH + calibrated SUMO | same, uncalibrated SUMO |
|---|---|---|---|---|---|
| US-101 | 18.3% | 35.8% | 0.7% | 15.5% | 40.6% |
| I-80 | 32.6% | 27.1% | 0.5% | 7.5% | 24.1% |
| Lankershim | 45.6% | 27.7% | 3.1% | 32.2% | 81.2% |
| Peachtree | 28.5% | 15.5% | 0.7% | 13.6% | 61.6% |

- with UAH as the only reference, dense US traffic has no matching situation (UAH has no jams, no freeways), the index falls back to all UAH normal windows, and in metres 18 to 46% of NGSIM windows look extreme. retrieval cannot invent a reference that does not exist.
- with headway and UAH as reference the shares drop to 0.5 to 3%.
- adding the SUMO planted normal drivers as the jam reference: uncalibrated, 24 to 81% flagged (SUMO normal drivers kept longer headways than real drivers); calibrated (tau x 0.7), 7.5 to 32%. calibration removes most of the gap; the rest follows from the speed distribution, which the lane drop network does not match (stop and go or free flow, never 48 km/h synchronized flow).

## explanation examples (`docs/hand_checks/rag_explanations.md`)

- UAH aggressive window (D6 motorway, raw 80.1): context score 100 among 200 normal motorway windows; proximity 48.8 of 80.1 points; 11.0 m at 110 km/h = 0.4 s headway against 2 s guidance (`knowledge/spain.md`); flagged.
- UAH normal window (median raw 29.3): context score 72; speed term largest, 94 km/h on a 90 km/h road.
- SUMO jam, normal driver (raw 68.8): context score 80 among normal jam windows; proximity 66.1 of 68.8 points from a 4.5 m gap at 6 km/h, which is a 2.5 s headway: in metres a stopped queue reads as tailgating, which is bug 4 again.

## bibliography J

`docs/bibliography/j_context_and_retrieval.md`: RAG-Driver (RSS 2024), MANTRA (CVPR 2020), Martinez et al. driving style survey (T-ITS 2018), Nasr Azadani and Boukerche driving behaviour analysis guidelines (T-ITS 2022).

## status

- [x] `Context` dataclass, built for UAH, SUMO, NGSIM and pNEUMA windows (`datasets/contexts.py`); the live SUMO agent scripts do not build it yet
- [x] window index (numpy, exact) instead of FAISS
- [x] context score = percentile among the k nearest normal windows of the matching situation
- [x] ablation: no context / 3 environments / RAG on UAH LODO and on the planted SUMO drivers
- [x] knowledge folder and an explanation that cites the file it used
- [x] 4 papers for bibliography J
- [ ] a labelled reference set from dense traffic (the missing piece for RAG on real traffic)
- [x] calibrate SUMO before using simulated windows as references (time headway; speed still off)
