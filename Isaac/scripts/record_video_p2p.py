"""Low-resolution video of a Play2Perfect policy on the union scene (RULES section 8), without Isaac Sim's renderer.

As ``record_video.py --renderer offline`` for Factory: PhysX runs the policy and every rigid body's pose is
recorded (``hetero_p2p/pose_record_p2p.py``), then ``render_offline.py --ground_z 0`` draws it with MuJoCo.
The env is set up as in ``eval_p2p.py`` (official evaluation overrides, deterministic actions); envs are
assigned hetero (env i on task i % 4), ``--per_task`` envs per task, one tile row per task.

    cd ~/play2perfect && ~/envs/p2p/bin/python ~/HeteroSimRL/Isaac/scripts/record_video_p2p.py --headless \\
        --checkpoint <run>/light/ep002000.pth --seconds 20 --out ~/runs/video/<name>
    MUJOCO_GL=egl ~/envs/render/bin/python ~/HeteroSimRL/Isaac/scripts/render_offline.py ~/runs/video/<name>.npz \\
        --ground_z 0 --eye 0.35 -0.35 0.3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))

parser = argparse.ArgumentParser(description="Record a Play2Perfect policy for offline rendering.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True, help="Output path without extension (.npz is added).")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--per_task", type=int, default=3)
parser.add_argument("--every", type=int, default=2, help="Record every this many policy steps (60 Hz policy).")
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
from hetero_p2p.pose_record_p2p import P2PPoseRecorder  # noqa: E402

REPO = Path(isaacsimenvs.__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.eval_isaacsim import _apply_env_overrides, _configure_agent  # noqa: E402

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
AGENT = "rl_games_sapg_cfg_entry_point"


def main():
    cfg = HeteroP2PEnvCfg()
    with open(REPO / "isaacsimenvs" / "cfg" / "task" / "PreciseAssembly.yaml") as f:
        overlay = yaml.safe_load(f) or {}
    overlay.pop("clip_observations", None)
    overlay.pop("clip_actions", None)
    cfg.from_dict(overlay)
    num_envs = args.per_task * len(cfg.problems)
    _apply_env_overrides(cfg, problem=cfg.problems[0], goal_mode="preInsertAndFinal", random_goal_fraction=0.0,
                         insertion_success_tolerance=0.01, retract_success_tolerance=0.005, num_envs=num_envs,
                         sim_device=args.sim_device, sdf=False, keep_dr=False, extra_overrides={})
    cfg.assignment.mode = "hetero"
    env = HeteroP2PEnv(cfg)
    agent_cfg = _configure_agent(TASK, AGENT, rl_device=args.rl_device, num_envs=num_envs, deterministic=True,
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
    player.reset()
    obs = player.env_reset(wrapped)

    recorder = P2PPoseRecorder(env)
    tiles = [[(int(e), name) for e in (env.task_ids == t).nonzero().flatten()[: args.per_task].tolist()]
             for t, name in enumerate(env.task_names)]
    steps = int(round(args.seconds / env.step_dt))
    t0 = time.perf_counter()
    for step in range(steps):
        action = player.get_action(obs, is_deterministic=True)
        obs, _rew, _dones, _infos = player.env_step(wrapped, action)
        if step % args.every == 0:
            recorder.record((step + 1) * env.step_dt)
    path = recorder.save(args.out, extra=dict(tiles=tiles, policy=args.checkpoint, dt=float(env.step_dt) * args.every,
                                              record_time_s=time.perf_counter() - t0))
    print(f"[INFO] wrote {path} ({len(recorder.time)} frames)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
