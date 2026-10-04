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

SUMO planted drivers, split by seed (reference seeds 0 to 4, test 5 to 9 and the reverse), AUC aggressive vs normal (`data/rag_ablation_sumo.csv`):

| | no context | no context, headway | 3 environments (agent) | RAG | RAG, headway |
|---|---|---|---|---|---|
| pooled over all scenarios | 0.751 | 0.830 | 0.730 | 0.778 | 0.849 |
| highway low | 0.897 | 0.891 | 0.876 | 0.900 | 0.895 |
| highway medium | 0.834 | 0.875 | 0.744 | 0.834 | 0.875 |
| jam | 0.714 | 0.785 | 0.658 | 0.729 | 0.790 |
| merge | 0.752 | 0.862 | 0.679 | 0.755 | 0.859 |
| roundabout | 0.916 | 0.981 | 0.886 | 0.915 | 0.977 |
| urban | 0.876 | 0.932 | 0.864 | 0.872 | 0.927 |
| weather | 0.822 | 0.869 | 0.838 | 0.821 | 0.869 |

reading:
- inside one situation a percentile is a monotone rescaling of the raw score, so RAG cannot change the AUC much there (per scenario rows). it helps only where situations are mixed: pooled SUMO +0.027 (metres) and +0.019 (headway).
- on UAH, RAG loses 0.028: UAH has two road types, no traffic state and free flow only, so there is little context to use, and many aggressive windows sit at the 100th percentile together (ties lose ranking information).
- measurement beats context: headway alone adds +0.079 on pooled SUMO, RAG adds +0.019 on top. best combination: headway + RAG, 0.849.
- the per environment agent is the weakest everywhere except weather: three weight sets learned from trip labels do not carry the situation.

## NGSIM: what the reference set does (no labels, `data/rag_ngsim_flags.csv`, `data/rag_ngsim_flags_headway.csv`)

share of NGSIM windows flagged (context score >= 95), against the raw cut off 70:

| site | raw >= 70, metres | RAG metres, ref UAH | RAG metres, ref UAH + SUMO | raw >= 70, headway | RAG headway, ref UAH | RAG headway, ref UAH + SUMO |
|---|---|---|---|---|---|---|
| US-101 | 10.2% | 18.3% | 24.3% | 3.1% | 0.7% | 40.6% |
| I-80 | 20.7% | 32.6% | 31.4% | 1.3% | 0.5% | 24.1% |
| Lankershim | 31.1% | 60.7% | 45.8% | 5.1% | 12.4% | 81.2% |
| Peachtree | 15.8% | 38.6% | 26.2% | 1.5% | 3.7% | 61.6% |

- with UAH as the only reference, dense US traffic has no matching situation (UAH has no jams, no freeways), the index falls back to all UAH normal windows, and in metres 18 to 61% of NGSIM windows look extreme. retrieval cannot invent a reference that does not exist.
- with headway and UAH as reference the shares drop to 0.5 to 12%.
- adding the SUMO planted normal drivers as a jam reference makes it worse (24 to 81% with headway): SUMO normal drivers keep longer headways than real US-101 drivers (part 2 calibration: NGSIM headways peak at 1.0 s, simulated ones at 1.4 s), so real traffic looks aggressive against them. an uncalibrated simulator is a bad reference set. calibrating tau to NGSIM (part 2, open) is a precondition for using simulated windows as references.

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
- [ ] calibrate SUMO before using simulated windows as references
