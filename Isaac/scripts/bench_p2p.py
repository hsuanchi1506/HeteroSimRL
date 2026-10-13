"""Alignment measurement on the Play2Perfect union scene (2026-10-12): does lining up the contact-rich work of all
envs in time save simulation time?

The idea only pays if a step costs about as much as its heaviest env (then steps with no heavy env are cheap);
if a step costs the sum of the envs' work (GPU full), the order does not matter. This measures both, in one
process (one scene build), with a trained policy:

* for each task mix in ``--configs``: set every env's task, reset every env at once (a synchronised start, i.e.
  aligned phases), run ``--steps`` policy steps; per step record the env-step time (GPU synchronised) and, per
  task, how many envs have their object touching its fixture (PhysX contact points between the two), how many
  hold the object lifted, and how many have reached each sub-goal. The first few hundred steps are the aligned
  regime; as episodes end at different times the phases spread out.
* the objects get PhysX's contact-report API for this (``object_contact_sensors``; off in training), which
  makes steps a little slower; ``--no_contacts`` measures without it.
* task mixes with 0, 1/16, 1/4, 1/2 and all envs on the heaviest task (screwing), the rest on the lightest
  (tight insertion): per-step time vs the number of heavy envs (max-like or sum-like).

Mixes run from light to heavy (PhysX grows its contact buffers and never shrinks them); the Hetero mix runs once
more at the end to see whether the grown buffers changed its time.

    cd ~/play2perfect && ~/envs/p2p/bin/python ~/HeteroSimRL/Isaac/scripts/bench_p2p.py --headless \\
        --checkpoint ~/runs/p2p_train/0_p2p_hetero/light/ep010000.pth --out ~/runs/p2p_bench/align.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))

DEFAULT_CONFIGS = [  # name: envs per task (tight_insertion, beam_assembly_step1, beam_assembly_step2, screwing), as fractions
    "homo_tight:1,0,0,0", "homo_beam2:0,0,1,0", "homo_beam1:0,1,0,0", "screw_1_16:15,0,0,1", "hetero:1,1,1,1",
    "screw_1_4:3,0,0,1", "screw_1_2:1,0,0,1", "homo_screw:0,0,0,1", "hetero_again:1,1,1,1",
]

parser = argparse.ArgumentParser(description="Alignment measurement on the Play2Perfect union scene.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=12288)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, help="name:w0,w1,w2,w3 (relative env counts)")
parser.add_argument("--stochastic", action="store_true", help="Sample actions (default: the mean, as evaluation).")
parser.add_argument("--no_contacts", action="store_true")
parser.add_argument("--rl_device", default="cuda:0")
parser.add_argument("--sim_device", default="cuda:0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

import isaacsimenvs  # noqa: E402, F401
from isaacsimenvs.utils.rlgames_utils import register_rlgames_env  # noqa: E402
from rl_games.torch_runner import Runner, _load_checkpoint_weights  # noqa: E402

from hetero_p2p.p2p_env import HeteroP2PEnv  # noqa: E402
from hetero_p2p.p2p_env_cfg import HeteroP2PEnvCfg  # noqa: E402

REPO = Path(isaacsimenvs.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.eval_isaacsim import _configure_agent  # noqa: E402


def interleaved(weights, n: int, device) -> torch.Tensor:
    """Task of every env for envs-per-task proportional to ``weights``, the tasks spread evenly over env ids."""
    w = [float(x) for x in weights]
    counts = [int(round(n * x / sum(w))) for x in w]
    counts[max(range(len(w)), key=lambda i: w[i])] += n - sum(counts)
    keys = sorted(((k + 0.5) / c, t) for t, c in enumerate(counts) for k in range(c))
    return torch.tensor([t for _, t in keys], dtype=torch.long, device=device), counts


class FixtureContacts:
    """Per env, the PhysX contact points between its active object and its active fixture (hole)."""

    def __init__(self, env, max_per_pair: int = 512):
        self.env = env

        def glob(path):
            return re.sub(r"/env_\d+/", "/env_*/", path)

        self.views, self.rows = [], []
        sim_view = env.robot._physics_sim_view
        for t in range(env.num_tasks):
            obj = env._objects[t].root_physx_view.prim_paths[0]
            hole = env._holes[t].root_physx_view.prim_paths[0]
            view = sim_view.create_rigid_contact_view(glob(obj), filter_patterns=[glob(hole)],
                                                      max_contact_data_count=max_per_pair * env.num_envs)
            if view is None:
                raise RuntimeError(f"no contact view for {glob(obj)} vs {glob(hole)}")
            env_of_row = [int(re.search(r"/env_(\d+)/", p).group(1)) for p in view.sensor_paths]
            rows = torch.empty(env.num_envs, dtype=torch.long, device=env.device)
            rows[torch.tensor(env_of_row, device=env.device)] = torch.arange(len(env_of_row), device=env.device)
            self.views.append(view)
            self.rows.append(rows)

    def points(self) -> torch.Tensor:
        env = self.env
        pts = torch.zeros(env.num_envs, device=env.device)
        for t, (view, rows) in enumerate(zip(self.views, self.rows)):
            ids = torch.nonzero(env.task_ids == t, as_tuple=False).reshape(-1)
            if ids.numel():
                counts = view.get_contact_data(dt=env.physics_dt)[4].reshape(view.sensor_count, view.filter_count)
                pts[ids] = counts[rows[ids], 0].float().to(env.device)
        return pts


def main():
    cfg = HeteroP2PEnvCfg()
    with open(REPO / "isaacsimenvs" / "cfg" / "task" / "PreciseAssembly.yaml") as f:
        cfg.from_dict(yaml.safe_load(f) or {})
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.sim_device
    cfg.assignment.mode = "hetero"
    cfg.object_contact_sensors = not args.no_contacts
    t0 = time.perf_counter()
    env = HeteroP2PEnv(cfg)
    build_s = time.perf_counter() - t0
    agent_cfg = _configure_agent("Isaacsimenvs-PreciseAssembly-Direct-v0", "rl_games_sapg_cfg_entry_point",
                                 rl_device=args.rl_device, num_envs=env.num_envs, deterministic=not args.stochastic,
                                 games=10**9, extra_overrides={})
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions)
    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.set_weights(_load_checkpoint_weights(player, args.checkpoint))
    player.has_batch_dimension = True
    contacts = None if args.no_contacts else FixtureContacts(env)
    all_ids = torch.arange(env.num_envs, device=env.device)
    out = dict(checkpoint=args.checkpoint, num_envs=env.num_envs, steps=args.steps, stochastic=args.stochastic,
               tasks=list(env.task_names), scene_build_s=build_s, configs={})
    for spec in args.configs:
        name, w = spec.split(":")
        tids, counts = interleaved(w.split(","), env.num_envs, env.device)
        env._set_task_rows(all_ids, tids)
        player.reset()
        obs = player.env_reset(wrapped)  # every env resets now: aligned phases
        rec = dict(counts=counts, step_ms=[], contact_envs=[], contact_pts=[], lifted=[], subgoal1=[], resets=[])
        tc = time.perf_counter()
        for _ in range(args.steps):
            action = player.get_action(obs, is_deterministic=not args.stochastic)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            obs, _rew, dones, _infos = player.env_step(wrapped, action)
            torch.cuda.synchronize()
            rec["step_ms"].append((time.perf_counter() - ts) * 1e3)
            rec["resets"].append(int(dones.reshape(-1).bool().sum()))
            per_t = lambda x: [int(x[env.task_ids == t].sum()) for t in range(env.num_tasks)]  # noqa: E731
            if contacts is not None:
                pts = contacts.points()
                rec["contact_envs"].append(per_t(pts > 0))
                rec["contact_pts"].append([round(float(pts[env.task_ids == t].sum()), 1) for t in range(env.num_tasks)])
            rec["lifted"].append(per_t(env._lifted_object))
            rec["subgoal1"].append(per_t(env._successes >= 1))
        rec["wall_s"] = time.perf_counter() - tc
        s = sorted(rec["step_ms"])
        print(f"[BENCH] {name} counts={counts} median {s[len(s) // 2]:.1f} ms  first-100 "
              f"{sum(rec['step_ms'][:100]) / 100:.1f} ms  last-300 {sum(rec['step_ms'][-300:]) / 300:.1f} ms", flush=True)
        out["configs"][name] = rec
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
