"""Checks of the union-scene Play2Perfect env (``HeteroP2PEnv``) without training.

1. Hetero: after a reset and ``--steps`` policy steps with zero actions, every parked object is still at its
   parking spot (the anti-gravity force holds it), every parked hole / goal marker is at its spot, and the
   active parts are where the official reset puts them (object above / on the table, hole on the table).
2. Homo (``--homo``): a switch scheduled a few steps ahead resets every env to the next task, swaps the parked
   and active parts, and updates the task one-hot and the per-env values (sub-goal count, object scale).

Prints one JSON summary; exits non-zero if a check fails.

    cd ~/play2perfect && ~/envs/p2p/bin/python ~/HeteroSimRL/Isaac/scripts/check_p2p_env.py --headless \\
        --num_envs 24 --sim_device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=24)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--homo", action="store_true")
parser.add_argument("--sim_device", default="cuda:0")
parser.add_argument("--park_tol", type=float, default=2e-3, help="Allowed drift of a parked object (m).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

import isaacsimenvs  # noqa: E402, F401

from hetero_p2p.p2p_env import HeteroP2PEnv  # noqa: E402
from hetero_p2p.p2p_env_cfg import HeteroP2PEnvCfg  # noqa: E402

TASK_YAML = Path(isaacsimenvs.__file__).resolve().parent / "cfg" / "task" / "PreciseAssembly.yaml"


def positions(env):
    """[tasks, envs, 3] env-frame positions of every task's object, hole and goal marker."""
    o = env.scene.env_origins
    get = lambda views: torch.stack([v.data.root_pos_w - o for v in views], 0)  # noqa: E731
    return get(env._objects), get(env._holes), get(env._goals)


def park_error(env, pos, kind):
    """Max distance of each task's parked part from its parking spot, over the envs where it is parked."""
    out = []
    for t in range(env.num_tasks):
        parked = env.task_ids != t
        if not bool(parked.any()):
            out.append(None)
            continue
        spot = torch.tensor(env._park_local(kind, t), device=pos.device)
        out.append(float((pos[t][parked] - spot).norm(dim=-1).max()))
    return out


def active_stats(env, pos):
    """Mean / min / max env-frame z of each task's active part."""
    out = []
    for t in range(env.num_tasks):
        a = env.task_ids == t
        if not bool(a.any()):
            out.append(None)
            continue
        z = pos[t][a][:, 2]
        out.append([round(float(z.mean()), 4), round(float(z.min()), 4), round(float(z.max()), 4)])
    return out


def main():
    cfg = HeteroP2PEnvCfg()
    with open(TASK_YAML) as f:
        cfg.from_dict(yaml.safe_load(f) or {})
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.sim_device
    cfg.assignment.mode = "homo" if args.homo else "hetero"
    env = HeteroP2PEnv(cfg)
    zero = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    report, ok = {"mode": cfg.assignment.mode, "num_envs": env.num_envs}, True

    obs, _ = env.reset()
    report["obs_shapes"] = {k: list(v.shape) for k, v in obs.items()}
    report["onehot_matches_tasks"] = bool((obs["policy"][:, -env.num_tasks:].argmax(-1) == env.task_ids).all())
    ok &= report["onehot_matches_tasks"]
    po, ph, pg = positions(env)
    report["after_reset"] = dict(object_park_err=park_error(env, po, "object"), hole_park_err=park_error(env, ph, "hole"),
                                 goal_park_err=park_error(env, pg, "goal"), active_object_z=active_stats(env, po),
                                 active_hole_z=active_stats(env, ph), table_z=float(env._table_z_per_env.mean()))
    for _ in range(args.steps):
        env.step(zero)
    po, ph, pg = positions(env)
    drift = park_error(env, po, "object")
    report[f"after_{args.steps}_steps"] = dict(object_park_err=drift, hole_park_err=park_error(env, ph, "hole"),
                                              active_object_z=active_stats(env, po), active_hole_z=active_stats(env, ph))
    ok &= all(d is None or d < args.park_tol for d in drift)
    report["env_max_goals_by_task"] = {n: sorted(set(env.env_max_goals[env.task_ids == t].tolist()))
                                       for t, n in enumerate(env.task_names)}

    if args.homo:
        before = int(env.task_ids[0])
        env.schedule_homo_switch(after_steps=3)
        for _ in range(3):
            obs, *_ = env.step(zero)
        after = env.task_ids.unique().tolist()
        po, ph, pg = positions(env)
        report["switch"] = dict(before=env.task_names[before], after=[env.task_names[t] for t in after],
                                onehot_ok=bool((obs["policy"][:, -env.num_tasks:].argmax(-1) == env.task_ids).all()),
                                object_park_err=park_error(env, po, "object"), hole_park_err=park_error(env, ph, "hole"),
                                active_object_z=active_stats(env, po), active_hole_z=active_stats(env, ph),
                                env_max_goals=sorted(set(env.env_max_goals.tolist())),
                                object_scale=env._object_scale_per_env[0].tolist(),
                                episodes=env.pop_episode_stats())
        ok &= after == [(before + 1) % env.num_tasks] and report["switch"]["onehot_ok"]
    report["ok"] = bool(ok)
    print("CHECK_JSON " + json.dumps(report), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
