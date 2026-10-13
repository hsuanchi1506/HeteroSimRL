"""Per-task success of a Homo / Hetero Play2Perfect policy: the official ``evaluation/eval_offline.py`` worker on
the union scene.

As the official worker: the official env config with the task YAML, then the official evaluation overrides
(``eval_isaacsim._apply_env_overrides``: randomisation off, ``force_consecutive_near_goal_steps``); the official
SAPG player, deterministic; every env's FIRST episode is one trial; trials that end within
``--min_episode_steps`` are dropped as unstable starts; success = every sub-goal reached. Differences: one process
holds all four tasks (``--per_task`` envs each, env i on task i % 4, so the policy sees its task one-hot), and
the checkpoint may be a light one written by ``train_p2p.py`` (weights only).

    cd ~/play2perfect && ~/envs/p2p/bin/python ~/HeteroSimRL/Isaac/scripts/eval_p2p.py --headless \\
        --checkpoint ~/runs/p2p_train/<run>/light/ep002000.pth --out ~/runs/p2p_eval/<run>_ep2000.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))

parser = argparse.ArgumentParser(description="Per-task offline evaluation on the union scene.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--per_task", type=int, default=1024, help="Envs (trials) per task; official: 1024.")
parser.add_argument("--steps", type=int, default=2500, help="Official default.")
parser.add_argument("--min_episode_steps", type=int, default=100, help="Official default.")
parser.add_argument("--goal_mode", default="preInsertAndFinal")
parser.add_argument("--problems", nargs="+", default=None)
parser.add_argument("--rl_device", default="cuda:0")
parser.add_argument("--sim_device", default="cuda:0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
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
from evaluation.eval_isaacsim import _apply_env_overrides, _configure_agent  # noqa: E402

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
AGENT = "rl_games_sapg_cfg_entry_point"


def main():
    t0 = time.perf_counter()
    cfg = HeteroP2PEnvCfg()
    with open(REPO / "isaacsimenvs" / "cfg" / "task" / "PreciseAssembly.yaml") as f:
        overlay = yaml.safe_load(f) or {}
    overlay.pop("clip_observations", None)  # official _load_env_cfg
    overlay.pop("clip_actions", None)
    cfg.from_dict(overlay)
    if args.problems:
        cfg.problems = tuple(args.problems)
    n_tasks = len(cfg.problems)
    num_envs = args.per_task * n_tasks
    _apply_env_overrides(cfg, problem=cfg.problems[0], goal_mode=args.goal_mode, random_goal_fraction=0.0,
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

    # official worker loop
    n, dev = num_envs, env.device
    done_once = torch.zeros(n, dtype=torch.bool, device=dev)
    first_len = torch.zeros(n, dtype=torch.long, device=dev)
    first_full = torch.zeros(n, dtype=torch.bool, device=dev)
    first_ratio = torch.zeros(n, dtype=torch.float, device=dev)
    first_retract = torch.zeros(n, dtype=torch.bool, device=dev)
    task_of = env.task_ids.clone()  # fixed in hetero mode
    step = 0
    while step < args.steps and not bool(done_once.all()):
        action = player.get_action(obs, is_deterministic=True)
        obs, _rew, dones, _infos = player.env_step(wrapped, action)
        newly = dones.reshape(-1).to(dev).bool() & ~done_once
        ids = torch.nonzero(newly, as_tuple=False).reshape(-1)
        if ids.numel():
            succ = env._prev_episode_successes[ids]
            maxg = env.prev_episode_env_max_goals[ids].clamp_min(1)
            first_len[ids] = step + 1
            first_full[ids] = succ >= env.prev_episode_env_max_goals[ids]
            first_ratio[ids] = (succ.float() / maxg.float()).clamp(0.0, 1.0)
            rs = env.extras.get("episode_final", {}).get("retract_success")
            if rs is not None:
                first_retract[ids] = rs[ids].to(dev) > 0.5
            done_once[ids] = True
        step += 1

    valid = done_once & (first_len >= args.min_episode_steps)
    out = dict(checkpoint=args.checkpoint, per_task=args.per_task, steps_run=step, eval_time_s=None, tasks={})
    for t, name in enumerate(env.task_names):
        m = task_of == t
        v = valid & m
        nv = int(v.sum())
        out["tasks"][name] = dict(
            n_envs=int(m.sum()), n_first_episodes=nv, n_unstable_init=int((done_once & m).sum()) - nv,
            n_unfinished=int((~done_once & m).sum()),
            success_rate=float(first_full[v].float().mean()) if nv else 0.0,
            mean_goal_ratio=float(first_ratio[v].mean()) if nv else 0.0,
            retract_rate=float(first_retract[v].float().mean()) if nv else 0.0)
    out["eval_time_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
