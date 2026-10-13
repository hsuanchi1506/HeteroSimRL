"""Homo / Hetero training on the four Play2Perfect assembly tasks (RULES.md section 4, second task family).

Follows the official ``isaacsimenvs/train.py`` of Play2Perfect (commit ff2fc62): the official env config with
the task YAML ``cfg/task/PreciseAssembly.yaml`` laid over it, the official SAPG config
``cfg/train/PreciseAssemblySAPG.yaml``, ``register_rlgames_env``, the repository's own rl_games, stage 2 started
from the released play checkpoint. The released stage-2 checkpoints fix what the YAML leaves open: every one of
the four was trained with 12,288 envs in 6 SAPG blocks of 2,048 (their saved rollouts hold 12,288 envs whose
block column takes the 6 values 50, 40, ..., 0 on 2,048 envs each; the YAML says blocks of 4,096) and counts
epochs and frames from 0 (frame = epoch x 16 x 12,288), i.e. loaded the play weights without resuming its
counters (``--checkpoint_load_mode weights``). Those are the defaults here. What differs from the official run:

* the env is ``HeteroP2PEnv`` (union scene, four tasks, task one-hot; ``hetero_p2p/p2p_env.py``);
* the play checkpoint is widened for the one-hot first (``hetero_p2p/checkpoint_utils.py``);
* ``--epochs`` PPO epochs after the checkpoint (the official config runs until stopped: max_epochs 1e6);
* Homo switches task every ``--block`` epochs at a rollout boundary;
* metrics and light checkpoints (``hetero_p2p/train_utils.py``). No Hydra command line: every setting that
  differs from the two YAML files is an argument here and is written to run_info.json.

Run from the Play2Perfect repository with its environment, e.g.

    cd ~/play2perfect && ~/envs/p2p/bin/python ~/HeteroSimRL/Isaac/scripts/train_p2p.py --headless \\
        --mode hetero --epochs 2000 --checkpoint ~/runs/p2p_ckpt/play_onehot4.pth --save_every 250
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

parser = argparse.ArgumentParser(description="Homo / Hetero Play2Perfect training (SAPG, union scene).")
parser.add_argument("--mode", choices=["hetero", "homo"], required=True)
parser.add_argument("--num_envs", type=int, default=12288, help="As the released stage-2 checkpoints.")
parser.add_argument("--sapg_block_size", type=int, default=2048,
                    help="SAPG block size (expl_coef_block_size); 2048 as the released stage-2 checkpoints, so "
                         "12288 envs form the play checkpoint's 6 blocks. The YAML says 4096.")
parser.add_argument("--epochs", type=int, required=True, help="PPO epochs to run after the checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Default: the official config's seed (42).")
parser.add_argument("--block", type=int, default=40, help="Homo: switch task every this many epochs.")
parser.add_argument("--first_task", type=int, default=0, help="Homo: task of the first block.")
parser.add_argument("--problems", nargs="+", default=None, help="Default: the four released problems.")
parser.add_argument("--checkpoint", default=None, help="Widened play checkpoint (checkpoint_utils.py).")
parser.add_argument("--checkpoint_load_mode", choices=["resume", "weights"], default="weights",
                    help="weights: as the released stage-2 checkpoints (counters from 0, fresh optimiser).")
parser.add_argument("--run_name", default=None)
parser.add_argument("--log_root", default=str(Path.home() / "runs" / "p2p_train"))
parser.add_argument("--save_every", type=int, default=0, help="Light checkpoint every this many epochs (0: none).")
parser.add_argument("--no_sync_timing", action="store_true")
parser.add_argument("--rl_device", default="cuda:0")
parser.add_argument("--sim_device", default="cuda:0")
# smoke tests only (few envs): SAPG block size and minibatch that fit a tiny rollout
parser.add_argument("--smoke_block_size", type=int, default=None)
parser.add_argument("--smoke_minibatch", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaacsimenvs  # noqa: E402, F401  (official gym registrations)
from isaacsimenvs.utils.rlgames_utils import EnvStatsAlgoObserver, MultiObserver, register_rlgames_env  # noqa: E402

from hetero_factory.rl_utils import GpuSampler, StepTimer, software_versions  # noqa: E402
from hetero_p2p.p2p_env import HeteroP2PEnv  # noqa: E402
from hetero_p2p.p2p_env_cfg import HeteroP2PEnvCfg  # noqa: E402
from hetero_p2p.train_utils import P2PMetricsObserver, make_homo_rollout_hook, write_json  # noqa: E402

CFG_DIR = Path(isaacsimenvs.__file__).resolve().parent / "cfg"
TASK_YAML = CFG_DIR / "task" / "PreciseAssembly.yaml"
AGENT_YAML = CFG_DIR / "train" / "PreciseAssemblySAPG.yaml"


def main():
    t_launch = time.perf_counter()

    # ---- env config: official defaults, official task YAML, then ours
    env_cfg = HeteroP2PEnvCfg()
    with open(TASK_YAML) as f:
        env_cfg.from_dict(yaml.safe_load(f) or {})
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.sim_device
    if args.problems:
        env_cfg.problems = tuple(args.problems)
    env_cfg.assignment.mode = args.mode
    env_cfg.assignment.homo_task = args.first_task

    # ---- agent config: the official SAPG YAML (its Hydra interpolations resolved by hand)
    with open(AGENT_YAML) as f:
        agent_cfg = yaml.safe_load(f)
    pcfg = agent_cfg["params"]
    cfg = pcfg["config"]
    if args.seed is not None:
        pcfg["seed"] = args.seed
    env_cfg.seed = pcfg["seed"]
    cfg["num_actors"] = args.num_envs
    cfg["device"] = cfg["device_name"] = args.rl_device
    official = dict(max_epochs=cfg["max_epochs"], save_frequency=cfg["save_frequency"],
                    expl_coef_block_size=cfg["expl_coef_block_size"])
    cfg["expl_coef_block_size"] = args.sapg_block_size
    smoke = {}
    if args.smoke_block_size:
        smoke["expl_coef_block_size"] = cfg["expl_coef_block_size"] = args.smoke_block_size
    if args.smoke_minibatch:
        smoke["minibatch_size"] = cfg["minibatch_size"] = args.smoke_minibatch
        cfg["central_value_config"]["minibatch_size"] = args.smoke_minibatch

    start_epoch = 0
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        st = ck[0] if isinstance(ck, dict) and 0 in ck else ck
        if args.checkpoint_load_mode == "resume":
            start_epoch = int(st.get("epoch", 0))
        del ck, st
    cfg["max_epochs"] = start_epoch + args.epochs

    run_name = args.run_name or f"p2p_{args.mode}{'' if args.mode == 'hetero' else f'_b{args.block}'}_n{args.num_envs}_s{pcfg['seed']}"
    # the vendored rl_games reads a policy index from the "<int>_" prefix of the experiment name
    cfg["name"] = cfg["full_experiment_name"] = f"0_{run_name}"
    cfg["train_dir"] = args.log_root
    log_dir = os.path.join(args.log_root, cfg["name"])
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "agent.yaml"), "w") as f:
        yaml.safe_dump(agent_cfg, f)

    # ---- env
    t0 = time.perf_counter()
    env = HeteroP2PEnv(env_cfg)
    env_create_time = time.perf_counter() - t0
    block_size = int(cfg.get("expl_coef_block_size") or args.num_envs)
    env.set_block_size(block_size)
    timer = StepTimer(env, sync_cuda=not args.no_sync_timing)
    clip_obs = float(pcfg["env"].get("clip_observations", math.inf))
    clip_actions = float(pcfg["env"].get("clip_actions", math.inf))
    register_rlgames_env(timer, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    device_index = int(str(args.rl_device).split(":")[-1]) if ":" in str(args.rl_device) else 0
    gpu = GpuSampler(device_index)
    hook = None
    if args.mode == "homo":
        hook = make_homo_rollout_hook(env, args.block, int(cfg["horizon_length"]))
    info = dict(
        mode=args.mode, num_envs=args.num_envs, seed=pcfg["seed"], tasks=list(env.task_names),
        problem_table=env.table_problem.summary(), homo=dict(block=args.block, first_task=args.first_task)
        if args.mode == "homo" else None, epochs=args.epochs, start_epoch=start_epoch,
        max_epochs=cfg["max_epochs"], official=official, horizon_length=cfg["horizon_length"],
        sapg_block_size=block_size, env_steps_budget=args.epochs * args.num_envs * int(cfg["horizon_length"]),
        checkpoint=args.checkpoint, checkpoint_load_mode=args.checkpoint_load_mode, save_every=args.save_every,
        obs_dim=int(env.cfg.observation_space), state_dim=int(env.cfg.state_space),
        changes_from_official=[
            "task one-hot appended to the policy and critic inputs (RULES.md); play checkpoint widened with zero "
            "weights for it (hetero_p2p/checkpoint_utils.py)",
            "union scene: every env holds all four tasks' parts, unused ones parked (hetero_p2p/p2p_env_cfg.py)",
            f"max_epochs = checkpoint epoch + {args.epochs} (official: 1e6, i.e. until stopped)",
            f"SAPG block size {cfg['expl_coef_block_size']} (YAML: {official['expl_coef_block_size']}; the released "
            "stage-2 checkpoints: 2048)",
        ] + ([f"homo: every env runs one task; switch every {args.block} epochs at a rollout boundary"]
             if args.mode == "homo" else ["hetero: env i runs task i % num_tasks"]),
        smoke_overrides=smoke or None,
        task_yaml=str(TASK_YAML), agent_yaml=str(AGENT_YAML), gpu=gpu.name, driver=gpu.driver,
        versions=software_versions(), env_create_time_s=env_create_time, argv=sys.argv,
    )
    write_json(os.path.join(log_dir, "run_info.json"), info)

    # ---- train
    metrics = P2PMetricsObserver(log_dir, env, step_timer=timer, gpu=gpu, save_every=args.save_every,
                                 epoch_end_hook=hook)
    runner = Runner(MultiObserver([EnvStatsAlgoObserver(), metrics]))
    runner.load(agent_cfg)
    runner.reset()
    t_train = time.perf_counter()
    runner.run({"train": True, "play": False, "checkpoint": args.checkpoint,
                "checkpoint_load_mode": args.checkpoint_load_mode})
    info.update(train_time_s=time.perf_counter() - t_train, total_time_s=time.perf_counter() - t_launch)
    write_json(os.path.join(log_dir, "run_info.json"), info)
    gpu.close()


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except Exception:  # noqa: BLE001 - e.g. the collapse guard of train_utils.P2PMetricsObserver
        import traceback

        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)  # official: Kit shutdown can hang
