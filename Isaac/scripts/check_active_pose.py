"""Check that the one-read-per-role pose of the active parts equals the one-read-per-task version.

HeteroFactoryEnv reads each env's active part pose ("fixed" and "held") every physics step. The fast path
(cfg.fast_active_pose, default) makes one PhysX read per role over all tasks' slots; the old path makes one read
per task. This steps the env with random actions across an episode reset (so the task assignment changes) and
compares both every step; any difference fails.

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/check_active_pose.py --scene union --num_envs 12 --headless
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

parser = argparse.ArgumentParser(description="Fast vs per-task read of the active parts' poses.")
parser.add_argument("--scene", choices=["union", "list"], default="union")
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--homo_task", choices=["peg_insert", "gear_mesh", "nut_thread"], default=None)
parser.add_argument("--steps", type=int, default=500, help="Policy steps (the episode is 449 steps, so >= 1 reset).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401
from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.hetero_factory_list_env import HeteroFactoryListEnv  # noqa: E402
from hetero_factory.train_common import TASK, TASK_LIST  # noqa: E402


def main():
    task = TASK_LIST if args_cli.scene == "list" else TASK
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.assignment.mode = "hetero"
    if args_cli.homo_task:
        cfg.assignment.mode = "homo"
        cfg.assignment.homo_task = ["peg_insert", "gear_mesh", "nut_thread"].index(args_cli.homo_task)
    cfg.fast_active_pose = True
    env = (HeteroFactoryListEnv if args_cli.scene == "list" else HeteroFactoryEnv)(cfg, render_mode=None)
    assert env._role_rows is not None, "fast path not built"
    env.reset()
    gen = torch.Generator(device=env.device).manual_seed(0)
    worst, resets, task_sets = 0.0, 0, set()
    for step in range(args_cli.steps):
        a = torch.rand((env.num_envs, env.cfg.action_space), device=env.device, generator=gen) * 2 - 1
        _, _, term, trunc, _ = env.step(a)
        resets += int((term | trunc).any())
        task_sets.add(tuple(env.task_ids.tolist()))
        for role in ("fixed", "held"):
            fp, fq = env._active_root_pose(role)
            sp, sq = env._active_root_pose_by_task(role)
            worst = max(worst, float((fp - sp).abs().max()), float((fq - sq).abs().max()))
    ok = worst == 0.0
    print(f"CHECK_ACTIVE_POSE scene={args_cli.scene} n={env.num_envs} steps={args_cli.steps} resets={resets} "
          f"task_assignments_seen={len(task_sets)} max_abs_diff={worst:.3g} -> {'PASS' if ok else 'FAIL'}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
