"""Homo training: every env runs the same task, tasks alternate peg -> gear -> nut (RULES.md section 4.A).

The official episode (449 steps at 30 s) is longer than one PPO rollout (128 steps) and does not divide it, so
"every rollout holds one task" and "episodes are never cut" cannot both hold. ``--switch`` picks which to keep
(team decision pending; details in hetero_factory/train_common.py):

* ``rollout`` (default, the literal reading of RULES.md): switch every ``--block`` PPO iterations; the episode
  running at the switch is cut (as a time-out, value bootstrapped) and logged as "_truncated";
* ``episode``: switch after every ``--block`` complete episodes; the rollout across the switch holds two tasks.

Per cycle of three blocks each task gets the same number of samples (``samples_per_task`` in metrics.jsonl),
which is what RULES.md asks for the comparison with Hetero.

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_homo.py --num_envs 128 --switch rollout --block 4 --headless

Outputs in ``logs/rl_games/Factory/homo_<switch><block>_n<N>_s<seed>/``: as train_hetero.py. The gradient probe
also runs on the two updates around every rollout switch. Not run yet: written 2026-10-04.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/
from hetero_factory.train_common_args import add_common_args  # noqa: E402

TASKS = ("peg_insert", "gear_mesh", "nut_thread")

parser = argparse.ArgumentParser(description="Homo (one task at a time, alternating) Factory training with rl_games.")
add_common_args(parser)
parser.add_argument("--switch", choices=["rollout", "episode"], default="rollout",
                    help="Switch tasks at rollout boundaries (cuts episodes) or at episode boundaries (mixes rollouts).")
parser.add_argument("--block", type=int, default=4,
                    help="PPO iterations (rollout) or episodes (episode) per task before switching. Default 4: "
                         "4 rollouts x 128 steps = 512 steps, about one 449-step episode.")
parser.add_argument("--first_task", choices=TASKS, default="peg_insert", help="Task of the first block.")
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
from hetero_factory.train_common import TASK, run  # noqa: E402


@hydra_task_config(TASK, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    homo = dict(switch=args_cli.switch, block=args_cli.block, first_task=TASKS.index(args_cli.first_task))
    run(env_cfg, agent_cfg, args_cli, mode="homo", homo=homo)


if __name__ == "__main__":
    main()
    simulation_app.close()
