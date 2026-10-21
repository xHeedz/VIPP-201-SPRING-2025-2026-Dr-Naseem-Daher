# context for claude code on the workstation (vipp 301a, driver aggressiveness index)

paste this whole file as the first message, or save it as `CLAUDE.local.md` in the repo root on the workstation. fill in every `<FILL: ...>` first. the repo also has `CLAUDE.md` and `docs/next_session.md`: read both after this file.

## this machine (fill in)

- operating system: <FILL: e.g. Ubuntu 22.04 / Windows 11>
- cpu cores: <FILL> , ram: <FILL> GB , gpu: <FILL or "none needed">
- free disk where the data goes: <FILL> GB (need about 10 GB)
- python version: <FILL: `python --version`>
- repo path on this machine: <FILL: e.g. /home/hadi/VIPP-201-SPRING-2025-2026-Dr-Naseem-Daher>
- drivedna download path: <FILL: e.g. /data/drivedna>
- shared machine, anything not allowed (sudo, installs, other users' folders): <FILL>
- how results get back to the mac: <FILL: git push / usb / cloud folder>

## who and what

- Hadi Al Shmaissani, third year Electrical and Computer Engineering, American University of Beirut. supervisor: always "Dr. Daher" (Dr. Naseem Daher), never "Daher" alone.
- VIPP 301A (fall 2026). project: a closed form Aggressiveness Index (AI) that scores surrounding drivers from speed, acceleration, gap to the car ahead and lane offset. Hadi's part only computes the index; it never changes what the ego car does.
- repo: github.com/xHeedz/VIPP-201-SPRING-2025-2026-Dr-Naseem-Daher, branch main.

## how to work with Hadi

- he wants to learn: before each change say the file, the line, what the code does now and what it will do. show worked numbers. explain simply, no jargon without a plain definition.
- run commands yourself and report the real numbers, including bad ones.
- ask before git commit and git push.
- documents: no em or en dashes, no first or second person, no version numbers, lowercase notebook style headings, `[ ]` checkboxes, plain typography.

## the task on this machine: full DriveDNA

DriveDNA: 465 drivers, 4,121 drives, 10 Hz car signals (speed, radar gap to the car ahead, lane offsets). access granted on Hugging Face (account xHeedz). no aggressiveness labels: results are scores per driver, compared across drivers and drives.

steps (also in `docs/workstation_drivedna.md`):
- [ ] `git pull` (needs the 21 oct commit: `scripts/drivedna_full.py`, `datasets/cleaning.py`, `datasets/windows.py`, `data/reference_weights.json`)
- [ ] `python -m venv .venv`, activate, `pip install -r requirements.txt pyarrow huggingface_hub`
- [ ] `hf auth login`
- [ ] `hf download HenryYHW/DriveDNA --repo-type dataset --include "index/*" "raw_signals/*" --local-dir <drivedna path>` (about 7.3 GB; never videos/, embeddings/, features/)
- [ ] quick test: `python scripts/drivedna_full.py --root <drivedna path> --limit 20`
- [ ] full run: `python scripts/drivedna_full.py --root <drivedna path> --workers <cores>` (resumable, one file per drive)
- [ ] bring back `results/drivedna_full/drivedna_drivers.csv` and `drivedna_windows.csv.gz`

what the script does: human driving spans (`index/human_segments.parquet`) -> per second table (`datasets/drivedna.py` `per_second_frame`) -> cleaning (`datasets/cleaning.py`, same rules as every dataset) -> 10 s windows (`datasets/windows.py`) -> reference formula. tested on two sample drives through a mock of the release layout; the parquet reading itself was never run (the mac has no pyarrow), so watch the first 20 drives: check column names (`time_s`, `vEgo`, `leadOne_status`, `leadOne_dRel`, `laneLeft_y`, `laneRight_y`) and the `drives.parquet` car model column.

## the reference formula (21 oct 2026)

score = 100 x (0.25 speed + 0.35 accel² + 0.30 gap² + 0.10 lane) / (sum of the weights of the known terms)
- speed = km/h / 150; accel² = (|a| / 5)²; gap² = mix of (1 - metres / 50)² and (1 - headway / 3 s)², highway 0.5 / 0.5; lane = offset / 1.5 m
- unknown terms are left out and the other weights rescaled (missing data must not count as calm)
- labels: below 19 conservative, 19 to 27.5 normal, 27.5 or more aggressive
- code: `model/aggressiveness_model.py` `reference_terms`, `reference_score`, `reference_label`; weights in `data/reference_weights.json`

## where the project stands (21 oct 2026)

- cleaning: every dataset audited (`scripts/audit_datasets.py`); missing values used to be stored as 0 and read as calm driving
- formula tests: UAH (6 real drivers, tested on a driver never seen) AUC 0.84; simulated drivers aggressive vs normal 0.89, normal vs conservative 0.79
- supervised model (DynamicWeightAgent): AUC 0.87 measured on the margin (score minus its own environment threshold). never rank its raw scores across environments: that understated it (0.76 instead of 0.90) until 20 oct
- RL labeler (PPO, `env/labeling_ref.py`): 8 inputs + a "wait 5 s" action, 72% on the simulated drivers after about 30 s; DQN 72%; formula 75% after 60 s. PPO learned "no close car -> conservative"
- inverse RL: right for a group of drivers, too little information per single car
- full write-up: `docs/research_notebook/fall_2026_10_20_revision.md`; open items in `docs/next_session.md`

## open questions for Dr. Daher (do not decide alone)

- squared acceleration adds only 0.1 to 1.1 points: keep it, or count only hard braking?
- add speed relative to the speed limit or to the traffic (the missing information for freely driving cars)?
- simulated driver types: keep the realistic overlap or widen it?

## after the full DriveDNA run (if time allows)

- [ ] per driver: median score, share conservative / aggressive, points per term, spread across drives (is a driver consistent?)
- [ ] same driver in different cars: does the score follow the driver or the car?
- [ ] score by car ahead share (does following raise the score?)
- [ ] write a short notebook page `docs/research_notebook/fall_2026_10_22_drivedna_full.md` with the numbers
