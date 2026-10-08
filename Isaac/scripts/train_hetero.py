"""Hetero training: every PPO rollout holds all three Factory tasks (RULES.md section 4.B).

Tasks are split evenly over the envs; ``--layout`` picks interleaved or blocked env order. ``--scene`` picks
how the batch is stored:

* ``union`` (default): every env holds every task's parts, the unused ones parked; leftover envs rotate tasks
  every episode (hetero_factory_env.py);
* ``list``: the "big list", each env holds only its own task's parts; tasks are fixed per env, so leftover
  envs cannot rotate (hetero_factory_list_env.py).

Everything else (official rl_games flow and PPO config, the only allowed PPO change, metrics, gradient
diagnostics) is in ``hetero_factory/train_common.py``.

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_hetero.py --num_envs 128 --seed 0 --headless
    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_hetero.py --scene list --num_envs 128 --seed 0 --headless

Outputs in ``logs/rl_games/Factory/hetero[list]_n<N>_s<seed>/``: run_info.json, metrics.jsonl (per PPO
iteration), episodes.jsonl (per task per episode), grad_diag.jsonl, tensorboard. Not run yet: written 2026-10-04.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/
from hetero_factory.train_common_args import add_common_args  # noqa: E402

parser = argparse.ArgumentParser(description="Hetero (all tasks in every rollout) Factory training with rl_games.")
add_common_args(parser)
parser.add_argument("--layout", choices=["interleaved", "blocked"], default="interleaved", help="Env order of tasks.")
parser.add_argument("--scene", choices=["union", "list"], default="union",
                    help="union: every env holds every task's parts; list: each env only its own (big list).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs inside the launched app."""

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import hetero_factory  # noqa: E402, F401  (registers the task)
from hetero_factory.train_common import TASK, TASK_LIST, run  # noqa: E402

TASK_ID = TASK_LIST if args_cli.scene == "list" else TASK


@hydra_task_config(TASK_ID, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    run(env_cfg, agent_cfg, args_cli, mode="hetero", task=TASK_ID)


if __name__ == "__main__":
    main()
    simulation_app.close()
