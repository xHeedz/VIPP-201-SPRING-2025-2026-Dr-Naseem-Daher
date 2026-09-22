"""Central file locations, so every script reads and writes the same folders
regardless of the working directory it is launched from."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
FIG_DIR = os.path.join(ROOT, "results", "figures")
SLIDES_DIR = os.path.join(ROOT, "docs", "slides")
SUMO_DIR = os.path.join(ROOT, "env", "sumo_scenarios")

for _d in (DATA_DIR, LOG_DIR, FIG_DIR, SLIDES_DIR):
    os.makedirs(_d, exist_ok=True)


def sumo_binary_name(gui=False):
    """Return the SUMO executable name for the current platform."""
    name = "sumo-gui" if gui else "sumo"
    return name + ".exe" if os.name == "nt" else name
