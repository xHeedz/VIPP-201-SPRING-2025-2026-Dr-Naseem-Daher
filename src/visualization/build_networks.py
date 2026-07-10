"""
Run this ONCE before the agents to compile .nod.xml + .edg.xml -> .net.xml
Requires SUMO to be installed with netconvert in PATH (or SUMO_HOME set).
"""
import os
import sys
import subprocess
from shutil import which

BASE = os.path.dirname(os.path.abspath(__file__))


def find_netconvert():
    nc = which("netconvert") or which("netconvert.exe")
    if nc:
        return nc
    sumo_home = os.environ.get("SUMO_HOME", "")
    for candidate in [
        os.path.join(sumo_home, "bin", "netconvert"),
        os.path.join(sumo_home, "bin", "netconvert.exe"),
        r"C:\Program Files (x86)\Eclipse\Sumo\bin\netconvert.exe",
        r"C:\Program Files\Eclipse\Sumo\bin\netconvert.exe",
        r"C:\Program Files (x86)\Eclipse\SUMO\bin\netconvert.exe",
        r"C:\Program Files\Eclipse\SUMO\bin\netconvert.exe",
    ]:
        if os.path.exists(candidate):
            return candidate
    print("ERROR: netconvert not found. Install SUMO and add its bin/ to PATH.")
    sys.exit(1)


def build(name, folder, extra_flags=None):
    nc = find_netconvert()
    d = os.path.join(BASE, "sumo", folder)
    cmd = [
        nc,
        "--node-files",       os.path.join(d, f"{name}.nod.xml"),
        "--edge-files",       os.path.join(d, f"{name}.edg.xml"),
        "--output-file",      os.path.join(d, f"{name}.net.xml"),
        "--no-turnarounds",   "true",
        "--no-warnings",
    ]
    if extra_flags:
        cmd += extra_flags
    print(f"Building {name}.net.xml ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)
    print(f"  -> sumo/{folder}/{name}.net.xml created.")


if __name__ == "__main__":
    build("highway", "highway")
    build(
        "intersection",
        "intersection",
        extra_flags=[
            "--connection-files",
            os.path.join(BASE, "sumo", "intersection", "intersection.con.xml"),
        ],
    )
    print("\nAll networks built. You can now run sumo_highway_agent.py and sumo_urban_agent.py.")
