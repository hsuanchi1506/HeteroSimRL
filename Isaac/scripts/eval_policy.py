"""Evaluate a trained checkpoint on each Factory task separately, with deterministic actions.

Training curves use the stochastic training policy, mixed (Hetero) or alternating (Homo). For the final
comparison every checkpoint takes the same test: the union scene with every env on one task (Homo
assignment), ``--episodes`` full episodes per task, the policy's mean action. Success is the fraction of
envs that meet the task's success condition at the episode's last step (the official success check).

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/eval_policy.py --headless \\
        --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn/Factory.pth --out runs/eval/hetero_n128_s0.json
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

TASKS = ("peg_insert", "gear_mesh", "nut_thread")
parser = argparse.ArgumentParser(description="Deterministic per-task evaluation of a Factory checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="A .pth file, or a run's nn/ directory (then the last_*_ep_<N>_*.pth with the largest N).")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=2, help="Full episodes per task.")
parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of the mean action.")
parser.add_argument("--seed", type=int, default=1000, help="Env seed (different from the training seeds).")
parser.add_argument("--collision_stack_size", type=int, default=None)
parser.add_argument("--out", type=str, required=True, help="JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401
from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.train_common import TASK  # noqa: E402


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        cands = [c for c in p.glob("last_*_ep_*.pth") if re.search(r"_ep_(\d+)", c.name)]
        if not cands:
            raise SystemExit(f"no last_*_ep_*.pth in {p}")
        p = max(cands, key=lambda c: int(re.search(r"_ep_(\d+)", c.name).group(1)))
    return p


def main():
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.assignment.mode = "homo"
    env_cfg.assignment.homo_task = TASKS.index(args_cli.tasks[0])
    if args_cli.collision_stack_size is not None:
        env_cfg.sim.physx.gpu_collision_stack_size = args_cli.collision_stack_size
    het = HeteroFactoryEnv(env_cfg, render_mode=None)

    pcfg = agent_cfg["params"]
    env = RlGamesVecEnvWrapper(
        het, pcfg["config"]["device"], pcfg["env"].get("clip_observations", math.inf),
        pcfg["env"].get("clip_actions", math.inf), pcfg["env"].get("obs_groups"),
        pcfg["env"].get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    pcfg["load_checkpoint"] = True
    pcfg["load_path"] = str(resolve_checkpoint(args_cli.checkpoint))
    pcfg["config"]["num_actors"] = het.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(pcfg["load_path"])
    agent.reset()

    out = dict(checkpoint=pcfg["load_path"], num_envs=het.num_envs, episodes=args_cli.episodes,
               deterministic=not args_cli.stochastic, seed=args_cli.seed, episode_steps=int(het.max_episode_length),
               tasks={})
    t0 = time.perf_counter()
    for task in args_cli.tasks:
        het.set_homo_task(task)
        het.episode_log.clear()
        # everything in inference mode: the env's buffers become inference tensors during the steps, and a reset
        # outside inference mode may not update them in place
        with torch.inference_mode():
            obs = env.reset()  # the reset applies the task
            if isinstance(obs, dict):
                obs = obs["obs"]
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
            het.episode_log.clear()  # the reset above logs the previous (partial) episode
            records = []
            while len(records) < args_cli.episodes:
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=not args_cli.stochastic)
                obs, _, dones, _ = env.step(actions)
                if agent.is_rnn and agent.states is not None and len(dones) > 0:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
                records += [e for e in het.episode_log if e["task"] == task and not e["truncated"]]
                het.episode_log.clear()
        succ = [r["success"] for r in records[: args_cli.episodes]]
        rets = [r["ret"] for r in records[: args_cli.episodes]]
        out["tasks"][task] = dict(success=sum(succ) / len(succ), success_per_episode=succ,
                                  ret=sum(rets) / len(rets), n_envs=records[0]["n_envs"])
        print(f"[EVAL] {task}: success {out['tasks'][task]['success']:.3f} {succ} return {out['tasks'][task]['ret']:.1f}")
    out["eval_time_s"] = time.perf_counter() - t0
    Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("EVAL_REPORT " + json.dumps(out))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
