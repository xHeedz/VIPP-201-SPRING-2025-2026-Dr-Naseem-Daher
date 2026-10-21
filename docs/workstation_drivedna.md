# full drivedna on the workstation

same pipeline as every other source: human spans, per second table, cleaning (`datasets/cleaning.py`), 10 s windows, reference formula (`data/reference_weights.json`). script: `scripts/drivedna_full.py`. tested on two sample drives through a mock of the release layout (parquet read replaced by csv); the parquet read itself is untested because this mac has no pyarrow.

## setup

- [ ] clone the repo, `git pull` so `data/reference_weights.json` and the new scripts are there
- [ ] python 3.9 or newer: `python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt pyarrow huggingface_hub`
- [ ] `hf auth login` (account xHeedz, DriveDNA access granted)

## download (only what the script reads)

```
hf download HenryYHW/DriveDNA --repo-type dataset --include "index/*" "raw_signals/*" --local-dir /data/drivedna
```

raw_signals is about 7.3 GB. never `videos/`, `embeddings/`, `features/`.

## run

- [ ] quick test, 20 drives: `python scripts/drivedna_full.py --root /data/drivedna --limit 20`
- [ ] check `results/drivedna_full/drivedna_drivers.csv` has rows and the printed share of windows is not empty
- [ ] full run: `python scripts/drivedna_full.py --root /data/drivedna --workers 16`
- [ ] a stopped run continues where it stopped (one file per drive in `results/drivedna_full/windows/`)

## what comes back

- `results/drivedna_full/drivedna_windows.csv.gz`: every 10 s window, its terms, points per term, score, driver, drive, car model
- `results/drivedna_full/drivedna_drivers.csv`: per driver: windows, drives, car models, median score, share conservative / aggressive, mean points per term, car ahead share, spread of the drive medians (consistency of a driver across drives)

copy both back into `data/` on the mac for the analysis and the deck.

## if it fails

- missing column (for example `time_s`): the script falls back to row number / 10 for time; other missing signals raise an error per drive, printed and skipped
- `drives.parquet` without a car model column: car model left empty, everything else runs
- out of memory: lower `--workers`
