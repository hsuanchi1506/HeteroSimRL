"""First-run check of an env on the real Isaac Lab / PhysX: build it, reset, step, report.

Works for the official Factory tasks and both hetero scenes, so the same numbers can be compared:

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/smoke_env.py --task Isaac-Factory-PegInsert-Direct-v0 --num_envs 16 --headless
    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/smoke_env.py --task Isaac-HeteroFactory-Direct-v0 --num_envs 12 --headless
    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/smoke_env.py --task Isaac-HeteroFactoryList-Direct-v0 --num_envs 12 --headless

Prints (and writes to --out as JSON): build and reset time, observation sizes, NaN checks, mean reward,
per-task success and extras after the steps, step time (zero or random actions), and for the hetero scenes
where every part slot's rows are and how far each env's active parts are from the table (parked parts must
be far, active ones near). Not a timing benchmark: the GPU is shared and the step count is small.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

parser = argparse.ArgumentParser(description="Build, reset and step an env once.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--steps", type=int, default=60, help="Policy steps after the reset.")
parser.add_argument("--actions", choices=["zero", "random"], default="zero")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None, help="Write the report as JSON here.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401


def main():
    rep = dict(task=args_cli.task, num_envs=args_cli.num_envs, gpu=torch.cuda.get_device_name(0))
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    t0 = time.perf_counter()
    env = gym.make(args_cli.task, cfg=cfg)
    rep["build_s"] = time.perf_counter() - t0
    u = env.unwrapped

    t0 = time.perf_counter()
    obs, extras = env.reset()
    torch.cuda.synchronize()
    rep["reset_s"] = time.perf_counter() - t0
    rep["policy_obs"] = list(obs["policy"].shape)
    rep["critic_obs"] = list(obs["critic"].shape) if "critic" in obs else None
    rep["obs_finite_after_reset"] = bool(torch.isfinite(obs["policy"]).all())

    hetero = hasattr(u, "task_ids")
    if hetero:
        rep["task_ids"] = u.task_ids.tolist()
        rep["slot_rows"] = {s: u._slot_count(s) for s in u._slots}
        # active parts near the table (x ~ 0.6 in the env frame); every slot's parts, per env, in the env frame
        rep["fixed_pos_env"] = [[round(v, 4) for v in p] for p in u.fixed_pos.tolist()]
        rep["held_pos_env"] = [[round(v, 4) for v in p] for p in u.held_pos.tolist()]
        rep["held_to_fingertip_m"] = [round(v, 4) for v in (u.held_pos - u.fingertip_midpoint_pos).norm(dim=-1).tolist()]

    n = u.num_envs
    rew_sum = torch.zeros(n, device=u.device)
    finite = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args_cli.steps):
        if args_cli.actions == "zero":
            act = torch.zeros((n, u.cfg.action_space), device=u.device)
        else:
            act = torch.rand((n, u.cfg.action_space), device=u.device) * 2 - 1
        obs, rew, term, trunc, extras = env.step(act)
        rew_sum += rew
        finite &= bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    rep["steps"] = args_cli.steps
    rep["step_ms"] = 1000 * dt / args_cli.steps
    rep["all_finite"] = finite
    rep["mean_reward_per_step"] = float(rew_sum.mean() / args_cli.steps)
    rep["extras"] = {k: (float(v) if (torch.is_tensor(v) and v.dim() == 0) or isinstance(v, (int, float)) else str(type(v)))
                     for k, v in extras.items()}
    if hetero:
        rep["held_to_fingertip_after_m"] = [
            round(v, 4) for v in (u.held_pos - u.fingertip_midpoint_pos).norm(dim=-1).tolist()
        ]
    print("SMOKE_REPORT " + json.dumps(rep))
    if args_cli.out:
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
