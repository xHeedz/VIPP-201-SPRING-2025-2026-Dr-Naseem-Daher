"""
Runs the whole pipeline, in order, with one command.

    python main.py                   stages check to replay
    python main.py --only train      one stage (several allowed: --only train noise)
    python main.py --skip replay     everything except the SUMO window
    python main.py --experiments     also re-run the two synthetic SUMO experiments
    python main.py --no-gui          replay without the SUMO window (faster, nothing to watch)
    python main.py --only replay --color-by absolute    replay coloured by the index alone

This file does no work of its own. Each stage runs one script from scripts/,
exactly as it would be typed in Terminal, and prints that command first:

  check    Python, packages, SUMO and the data folders are all there
  test     python -m pytest tests
  train    scripts/train_uah.py      train and test the agent on real drivers (UAH)
  noise    scripts/noise_uah.py      re-test it on the held-out drivers through noisy sensors
  sample   cut the first minutes of one NGSIM location into a small file (only once)
  score    scripts/score_ngsim.py    apply it to real US traffic: index, shockwave, noise
  replay   scripts/ngsim_replay_sumo.py --gui --ellipses   watch that traffic live in SUMO

  experiments (off unless --experiments)
           scripts/noise_robustness.py and scripts/shockwave_experiment.py,
           the earlier experiments on synthetic SUMO traffic

Setting up .venv and installing packages is not a stage: this file needs those
packages before it can run. Committing and pushing to GitHub is not a stage either.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- where the data lives
# The data sits next to the repository, in the folder "28:9:2026", and never
# goes into the repository itself. Change these lines if it moves.
DATA_FOLDER = os.path.abspath(os.path.join(ROOT, "..", "28:9:2026"))
UAH_ROOT = os.path.join(DATA_FOLDER, "UAH-DRIVESET-v1")
NGSIM_FILE = os.path.join(DATA_FOLDER,
                          "Next_Generation_Simulation_(NGSIM)_Vehicle_Trajectories_and_Supporting_Data_20260921.csv")
LOCATION = "us-101"          # us-101, i-80, lankershim or peachtree
MINUTES = 5.0                # how much of that location to use
NGSIM_SAMPLE = os.path.join(DATA_FOLDER, f"ngsim_{LOCATION}_{MINUTES:g}min.csv")

STAGES = ["check", "test", "train", "noise", "sample", "score", "replay"]
PACKAGES = ["numpy", "pandas", "matplotlib", "torch", "sumo", "traci", "pytest"]


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78, flush=True)


def run(script_and_args):
    """Run one command in the repository folder, the same way it would be typed in Terminal."""
    cmd = [sys.executable] + script_and_args
    shown = " ".join(f'"{c}"' if " " in c else c for c in ["python"] + script_and_args)
    print(f"$ {shown}", flush=True)
    start = time.time()
    code = subprocess.call(cmd, cwd=ROOT, env=os.environ.copy())
    if code != 0:
        sys.exit(f"\nstopped: that command failed (exit code {code}). The error is printed above it.")
    print(f"done in {time.time() - start:.0f} s", flush=True)


# ---------------------------------------------------------------- the stages
def check():
    """Everything the later stages need. Also finds SUMO, so SUMO_HOME never has to be typed."""
    problems = []
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    if ".venv" not in sys.executable:
        print("  note: this is not the .venv Python; run 'source .venv/bin/activate' first")
    for name in PACKAGES:
        try:
            __import__(name)
            print(f"  ok   {name}")
        except ImportError:
            problems.append(f"package {name} is missing (pip install {'eclipse-sumo' if name == 'sumo' else name})")
    try:
        import sumo
        os.environ["SUMO_HOME"] = sumo.SUMO_HOME          # every later stage inherits this
        print(f"  SUMO_HOME = {sumo.SUMO_HOME}")
        for tool in ("sumo", "sumo-gui", "netconvert"):
            found = os.path.exists(os.path.join(sumo.SUMO_HOME, "bin", tool))
            print(f"  {'ok  ' if found else 'MISSING'} SUMO program {tool}")
            if not found and tool != "sumo-gui":
                problems.append(f"SUMO tool {tool} is missing")
    except ImportError:
        pass
    for label, path in (("UAH folder", UAH_ROOT), ("NGSIM file", NGSIM_FILE)):
        ok = os.path.exists(path)
        print(f"  {'ok  ' if ok else 'MISSING'} {label}: {path}")
        if not ok and not (label == "NGSIM file" and os.path.exists(NGSIM_SAMPLE)):
            problems.append(f"{label} not found at {path}")
    if problems:
        sys.exit("\nstopped:\n  " + "\n  ".join(problems))


def test():
    run(["-m", "pytest", "tests", "-q"])


def train():
    run(["scripts/train_uah.py", "--root", UAH_ROOT])


def noise():
    run(["scripts/noise_uah.py", "--root", UAH_ROOT])


def sample():
    """Reads the large NGSIM file once and keeps only the first MINUTES of LOCATION."""
    if os.path.exists(NGSIM_SAMPLE):
        print(f"already there: {NGSIM_SAMPLE}")
        return
    print(f"reading {os.path.basename(NGSIM_FILE)} (about 2 GB, a few minutes)...", flush=True)
    sys.path.insert(0, ROOT)
    from datasets.ngsim import read_raw
    raw = read_raw(NGSIM_FILE, LOCATION, MINUTES)
    raw.drop(columns="t").to_csv(NGSIM_SAMPLE, index=False)
    print(f"wrote {len(raw):,} rows ({raw['Vehicle_ID'].nunique()} vehicles) to {NGSIM_SAMPLE}")


def score():
    run(["scripts/score_ngsim.py", "--file", NGSIM_SAMPLE, "--location", LOCATION, "--minutes", f"{MINUTES:g}"])


def replay(gui=True, color_by="context"):
    args = ["scripts/ngsim_replay_sumo.py", "--file", NGSIM_SAMPLE, "--location", LOCATION,
            "--minutes", f"{MINUTES:g}", "--ellipses", "--color-by", color_by]
    if gui and sys.platform == "darwin" and not os.path.exists("/Applications/Utilities/XQuartz.app"):
        # sumo-gui draws its window with X11, which macOS only has once XQuartz is installed
        print("XQuartz is not installed, so the SUMO window cannot open. To see it:\n"
              "  1. install XQuartz from https://www.xquartz.org\n"
              "  2. log out of the Mac and back in\n"
              "  3. python main.py --only replay\n"
              "running the replay without a window for now")
        gui = False
    if gui:
        print("the SUMO window opens; close it when finished to end this stage")
        args += ["--gui", "--delay", "0"]
    run(args)


def experiments():
    run(["scripts/noise_robustness.py"])
    run(["scripts/shockwave_experiment.py"])


# ---------------------------------------------------------------- choosing what runs
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the pipeline, stage by stage.")
    ap.add_argument("--only", nargs="+", choices=STAGES + ["experiments"], help="run only these stages")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[], help="leave these stages out")
    ap.add_argument("--experiments", action="store_true", help="also re-run the synthetic SUMO experiments")
    ap.add_argument("--no-gui", action="store_true", help="replay without opening the SUMO window")
    ap.add_argument("--color-by", choices=["context", "absolute", "rank"], default="context",
                    help="replay colours: relative to nearby traffic (default), the index alone, or display-only rank")
    a = ap.parse_args()

    chosen = a.only or [s for s in STAGES if s not in a.skip] + (["experiments"] if a.experiments else [])
    if "check" not in chosen:
        chosen = ["check"] + chosen          # always check first: it also sets SUMO_HOME
    steps = {"check": check, "test": test, "train": train, "noise": noise, "sample": sample,
             "score": score, "replay": lambda: replay(gui=not a.no_gui, color_by=a.color_by), "experiments": experiments}
    for i, name in enumerate(chosen, 1):
        banner(f"stage {i} of {len(chosen)}: {name}")
        steps[name]()
    banner("all stages finished")
