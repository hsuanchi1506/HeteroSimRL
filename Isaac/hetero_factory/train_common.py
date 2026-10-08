"""Shared body of the Homo and Hetero training scripts (``scripts/train_homo.py``, ``scripts/train_hetero.py``).

Follows Isaac Lab's official ``scripts/reinforcement_learning/rl_games/train.py`` (identical in v2.3.2 and
main): Hydra-loaded configs, RlGamesVecEnvWrapper, rl_games Runner, the official Factory rl_games config.
Import this module only after the Isaac Sim app has been launched.

Homo (RULES.md section 4.A): every env runs the same task; tasks cycle peg -> gear -> nut. Two ways to switch,
because the official episode (449 steps) is longer than a PPO rollout (128 steps) and does not divide it:

* ``rollout`` (default): switch every ``--block`` PPO iterations. On the last step of the block's last rollout
  every env times out (a truncation, so PPO bootstraps the value), and the reset starts the next task. Every
  rollout holds one task, as RULES.md asks; the episode running at the switch is cut short and logged as
  "_truncated".
* ``episode``: switch after every ``--block`` complete episodes. No episode is cut, but the rollout that spans
  the switch holds two tasks.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from .grad_probe import GradientProbe
from .hetero_factory_env_cfg import HETERO_OBS_DIM_CFG
from .rl_utils import (
    GpuSampler,
    MetricsObserver,
    StepTimer,
    epochs_for_env_steps,
    fix_minibatch,
    make_homo_rollout_hook,
    software_versions,
)

TASK = "Isaac-HeteroFactory-Direct-v0"  # union scene (every env holds every task's parts)
TASK_LIST = "Isaac-HeteroFactoryList-Direct-v0"  # big-list scene (each env holds only its own parts; hetero only)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def run(env_cfg, agent_cfg: dict, args, mode: str, homo: dict | None = None, task: str = TASK):
    """Train one run. ``mode`` is "hetero" or "homo"; ``homo`` = dict(switch="rollout"|"episode", block=int,
    first_task=int) for homo; hetero uses ``args.layout``. ``task`` is the gym id (union or big-list scene)."""
    if task == TASK_LIST and mode != "hetero":
        raise ValueError("the big-list scene is hetero only")
    t_launch = time.perf_counter()

    # ---- env config
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device if getattr(args, "device", None) is not None else env_cfg.sim.device
    env_cfg.assignment.mode = mode
    if mode == "hetero":
        env_cfg.assignment.layout = args.layout
    else:
        env_cfg.assignment.homo_task = int(homo["first_task"])
        env_cfg.assignment.homo_switch_every_episodes = int(homo["block"]) if homo["switch"] == "episode" else 0
    num_envs = env_cfg.scene.num_envs
    if getattr(args, "collision_stack_size", None) is not None:
        env_cfg.sim.physx.gpu_collision_stack_size = args.collision_stack_size

    # ---- agent config: official, plus the changes RULES.md allows
    pcfg = agent_cfg["params"]
    pcfg["seed"] = args.seed if args.seed is not None else pcfg["seed"]
    env_cfg.seed = pcfg["seed"]
    official_epochs = pcfg["config"]["max_epochs"]
    horizon = pcfg["config"]["horizon_length"]
    if args.env_steps is not None:
        pcfg["config"]["max_epochs"] = epochs_for_env_steps(args.env_steps, num_envs, horizon)
    elif args.max_iterations is not None:
        pcfg["config"]["max_epochs"] = args.max_iterations
    minibatch_changes = fix_minibatch(agent_cfg, num_envs)
    official_save_frequency = pcfg["config"].get("save_frequency")
    if args.save_every is not None:  # checkpoints for the periodic evaluation / videos of RULES section 8 (not PPO)
        pcfg["config"]["save_frequency"] = args.save_every
    if args.checkpoint is not None:
        pcfg["load_checkpoint"] = True
        pcfg["load_path"] = os.path.abspath(args.checkpoint)

    # ---- logging directory (as the official script: logs/rl_games/<config name>/<run>)
    if args.run_name:
        run_name = args.run_name
    elif mode == "hetero":
        run_name = f"hetero{'list' if task == TASK_LIST else ''}_n{num_envs}_s{pcfg['seed']}"
    else:
        run_name = f"homo_{homo['switch']}{homo['block']}_n{num_envs}_s{pcfg['seed']}"
    log_root = os.path.abspath(os.path.join("logs", "rl_games", pcfg["config"]["name"]))
    log_dir = os.path.join(log_root, run_name)
    pcfg["config"]["train_dir"] = log_root
    pcfg["config"]["full_experiment_name"] = run_name
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging to {log_dir}")

    # ---- environment
    rl_device = pcfg["config"]["device"]
    clip_obs = pcfg["env"].get("clip_observations", math.inf)
    clip_actions = pcfg["env"].get("clip_actions", math.inf)
    obs_groups = pcfg["env"].get("obs_groups")
    concate_obs_groups = pcfg["env"].get("concate_obs_groups", True)

    t0 = time.perf_counter()
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    env_create_time = time.perf_counter() - t0
    het = env.unwrapped
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    timer = StepTimer(env, sync_cuda=not args.no_sync_timing)
    env = RlGamesVecEnvWrapper(timer, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    pcfg["config"]["num_actors"] = het.num_envs

    # ---- run info
    device_index = int(str(rl_device).split(":")[-1]) if ":" in str(rl_device) else 0
    gpu = GpuSampler(device_index)
    info = dict(
        mode=mode, task=task, scene="list" if task == TASK_LIST else "union", num_envs=num_envs, seed=pcfg["seed"],
        layout=getattr(args, "layout", None) if mode == "hetero" else None,
        homo=homo if mode == "homo" else None,
        max_epochs=pcfg["config"]["max_epochs"], official_max_epochs=official_epochs, env_steps_budget=args.env_steps,
        horizon_length=horizon, episode_length_s=env_cfg.episode_length_s,
        save_frequency=pcfg["config"].get("save_frequency"), official_save_frequency=official_save_frequency,
        episode_length_mode=env_cfg.episode_length_mode, ppo_changes=minibatch_changes,
        task_names=list(het.task_names), gpu=gpu.name, driver=gpu.driver, nvml_ok=gpu.ok,
        gpu_collision_stack_size=env_cfg.sim.physx.gpu_collision_stack_size,
        versions=software_versions(), env_create_time_s=env_create_time, argv=sys.argv,
    )

    # ---- the task one-hot's columns in the policy input (gradient probe, per-task advantage statistics)
    start = 0
    for name in env_cfg.obs_order:
        if name == "task_onehot":
            break
        start += HETERO_OBS_DIM_CFG[name]
    onehot = slice(start, start + HETERO_OBS_DIM_CFG["task_onehot"])
    adv_norm = getattr(args, "adv_norm", "global")
    info["adv_norm"] = adv_norm
    if adv_norm == "per_task":
        info["ppo_changes"] = list(info["ppo_changes"]) + [dict(
            key="advantage normalisation", official="whole rollout (rl_games normalize_advantage)",
            used="per task (analysis experiment, hetero_factory.rl_utils.per_task_advantages)")]
    elif adv_norm == "pooled":
        info["ppo_changes"] = list(info["ppo_changes"]) + [dict(
            key="advantage normalisation", official="whole rollout (rl_games normalize_advantage)",
            used="pooled over the latest rollout of every task (2026-10-07 meeting TODO1; for Homo; equals the "
                 "official one for Hetero; hetero_factory.rl_utils.pooled_task_advantages)")]

    # ---- gradient diagnostics (RULES.md section 5)
    probe = None
    if args.grad_every > 0:
        probe = GradientProbe(log_dir, het.task_names, onehot, every=args.grad_every, chunk_seqs=args.grad_chunk_seqs,
                              max_seqs=args.grad_max_seqs, seed=pcfg["seed"])
        info["grad_probe"] = dict(every=args.grad_every, chunk_seqs=args.grad_chunk_seqs, max_seqs=args.grad_max_seqs,
                                  onehot_columns=[onehot.start, onehot.stop])

    # ---- homo switching at rollout boundaries
    hook = None
    if mode == "homo" and homo["switch"] == "rollout":
        hook = make_homo_rollout_hook(het, int(homo["block"]), horizon, probe)
        info["homo_switch_note"] = "the task changes after the last env step of every PPO iteration k*block"
    elif mode == "homo" and homo["switch"] == "episode":
        last = {"task": het.homo_task}

        def hook(epoch: int):
            # probe the first update after a task change (the change happened at an episode end in this epoch)
            if het.homo_task != last["task"]:
                last["task"] = het.homo_task
                if probe is not None:
                    probe.force_next(1)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _write_json(os.path.join(log_dir, "run_info.json"), info)
    print_dict(info, nesting=4)

    # ---- train
    observer = MetricsObserver(log_dir, het.task_names, step_timer=timer, gpu=gpu, warmup_epochs=args.warmup_epochs,
                               env=het, probe=probe, epoch_end_hook=hook, adv_onehot=onehot, adv_mode=adv_norm)
    runner = Runner(observer)
    runner.load(agent_cfg)
    runner.reset()
    t_train = time.perf_counter()
    if args.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": None, "checkpoint": pcfg["load_path"]})
    else:
        runner.run({"train": True, "play": False, "sigma": None})
    train_time = time.perf_counter() - t_train

    info.update(train_time_s=train_time, total_time_s=time.perf_counter() - t_launch,
                grad_diag_time_s=probe.time_total if probe is not None else 0.0)
    _write_json(os.path.join(log_dir, "run_info.json"), info)
    print(f"Training time: {train_time:.1f} s")
    gpu.close()
    env.close()
