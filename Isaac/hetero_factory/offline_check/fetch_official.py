"""Download the official Isaac Lab Factory source files that the offline checks load (not stored in this repo).

    python fetch_official.py [--ref v2.3.2]

The five files are identical in v2.3.2 and main (checked 2026-10-04).
"""
import argparse
import urllib.request
from pathlib import Path

FILES = ["factory_env.py", "factory_env_cfg.py", "factory_tasks_cfg.py", "factory_utils.py", "factory_control.py",
         "agents/rl_games_ppo_cfg.yaml"]
URL = "https://raw.githubusercontent.com/isaac-sim/IsaacLab/{ref}/source/isaaclab_tasks/isaaclab_tasks/direct/factory/{f}"

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="v2.3.2")
args = ap.parse_args()
dest = Path(__file__).resolve().parent / "stubs" / "isaaclab_tasks" / "direct" / "factory"
for f in FILES:
    with urllib.request.urlopen(URL.format(ref=args.ref, f=f), timeout=60) as r:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        (dest / f).write_bytes(r.read())
    print(f"{args.ref}: {f}")
