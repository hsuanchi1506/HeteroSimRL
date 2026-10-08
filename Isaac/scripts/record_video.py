"""Record a low-resolution video of the hetero Factory batch in Isaac Sim / PhysX (RULES.md section 8).

A few envs per task are tiled into a grid (one row per task); each tile looks at that env's fixed part, shows
the task, the env index and turns green while the env meets its task's success condition. The policy is a
trained rl_games checkpoint (loaded as in Isaac Lab's official ``scripts/reinforcement_learning/rl_games/play.py``),
or, before any training, zero or random actions so the scene itself can be checked: parts in place, held part
in the hand, the three tasks side by side.

Two ways to get pictures:

* ``--renderer offline`` (default): no camera in Isaac Sim. The run saves every body's pose per step and the
  meshes (``<out>.npz``, see hetero_factory/pose_record.py); ``scripts/render_offline.py`` then draws them with
  MuJoCo's OpenGL renderer. MuJoCo only draws: every pose comes from this PhysX run. Needed on gputw1, where
  Isaac Sim 5.1's RTX renderer segfaults with driver 595 (NVIDIA validates 5.1 with the 580 driver only).
* ``--renderer rtx``: one Isaac Lab TiledCamera per env (needs ``--enable_cameras`` and a validated driver).

Videos are for checking only, never recorded during timing runs (rendering changes the step time).

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/record_video.py --headless --policy zero --out runs/video/union_zero
    MUJOCO_GL=egl ~/envs/render/bin/python ../HeteroSimRL/Isaac/scripts/render_offline.py runs/video/union_zero.npz
    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/record_video.py --headless \\
        --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn/Factory.pth --scene union

Writes ``<out>.json`` (which env is in which tile, success per tile per step) and ``<out>.npz`` (offline) or
``<out>.mp4`` (rtx; PNG frames if no video writer is installed).
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

parser = argparse.ArgumentParser(description="Record a video of the hetero Factory batch.")
parser.add_argument("--scene", choices=["union", "list"], default="union", help="Union scene or big-list scene.")
parser.add_argument("--renderer", choices=["offline", "rtx"], default="offline",
                    help="offline: save poses, draw with scripts/render_offline.py; rtx: Isaac Sim cameras.")
parser.add_argument("--num_envs", type=int, default=12, help="Envs to simulate (split evenly over the tasks).")
parser.add_argument("--per_task", type=int, default=3, help="Tiles per task (one row per task).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="rl_games checkpoint (.pth, or a run's nn/ directory: its last_*_ep_<N>_*.pth with the largest "
                         "N); without it, --policy.")
parser.add_argument("--policy", choices=["zero", "random"], default="zero", help="Actions when no checkpoint.")
parser.add_argument("--seconds", type=float, default=None, help="Video length (default: one episode).")
parser.add_argument("--tile", type=int, nargs=2, default=[240, 180], help="Tile width and height in pixels.")
parser.add_argument("--eye", type=float, nargs=3, default=[0.25, -0.2, 0.17],
                    help="Camera position relative to the fixed part's tip (m, env frame).")
parser.add_argument("--seed", type=int, default=0, help="Seed.")
parser.add_argument("--out", type=str, default="videos/hetero_factory", help="Output path without extension.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.renderer == "rtx":
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs inside the launched app."""

import json  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401
from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.hetero_factory_env_cfg import PART_SLOTS  # noqa: E402
from hetero_factory.hetero_factory_list_env import HeteroFactoryListEnv  # noqa: E402
from hetero_factory.train_common import TASK, TASK_LIST  # noqa: E402
from hetero_factory.pose_record import PoseRecorder  # noqa: E402
from hetero_factory.video_utils import Writer, compose  # noqa: E402


def with_camera(base):
    """The env class plus one TiledCamera per env. Created in the scene's extra-prims hook, i.e. before the
    union scene clones env 0 (as Isaac Lab's official camera tasks do) and after the big-list scene has built
    every env."""

    class WithCamera(base):
        def _setup_extra_scene(self):
            super()._setup_extra_scene()
            self._video_camera = TiledCamera(self.cfg.video_camera)
            self.scene.sensors["video_camera"] = self._video_camera

    return WithCamera


def aim_cameras(env):
    """Point every env's camera at its fixed part's tip (called after each reset, the part moves)."""
    target = env.fixed_pos_obs_frame + env.scene.env_origins
    eye = target + torch.tensor(args_cli.eye, device=env.device)
    env._video_camera.set_world_poses_from_view(eye, target)


def make_policy(env_wrapped, agent_cfg, het):
    """rl_games player as in Isaac Lab's play.py, or a stand-in."""
    if args_cli.checkpoint is None:
        def act(obs):
            if args_cli.policy == "zero":
                return torch.zeros((het.num_envs, het.cfg.action_space), device=het.device)
            return torch.rand((het.num_envs, het.cfg.action_space), device=het.device) * 2 - 1
        return act, None

    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    from isaaclab_rl.rl_games import RlGamesGpuEnv

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env_wrapped})
    import re

    ck = Path(args_cli.checkpoint).expanduser().resolve()
    if ck.is_dir():
        ck = max(ck.glob("last_*_ep_*.pth"), key=lambda c: int(re.search(r"_ep_(\d+)", c.name).group(1)))
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(ck)
    agent_cfg["params"]["config"]["num_actors"] = het.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(agent_cfg["params"]["load_path"])
    agent.reset()
    return agent, agent


def main():
    task = TASK_LIST if args_cli.scene == "list" else TASK
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.assignment.mode = "hetero"
    rtx = args_cli.renderer == "rtx"
    base = HeteroFactoryListEnv if args_cli.scene == "list" else HeteroFactoryEnv
    if rtx:
        env_cfg.sim.render_interval = env_cfg.decimation  # render once per control step
        w, h = args_cli.tile
        env_cfg.video_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/VideoCamera",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.85, -0.2, 0.25), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)
            ),
            width=w,
            height=h,
        )
        het = with_camera(base)(env_cfg, render_mode=None)
    else:
        het = base(env_cfg, render_mode=None)

    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    pcfg = agent_cfg["params"]
    env = RlGamesVecEnvWrapper(
        het, pcfg["config"]["device"], pcfg["env"].get("clip_observations", math.inf),
        pcfg["env"].get("clip_actions", math.inf), pcfg["env"].get("obs_groups"),
        pcfg["env"].get("concate_obs_groups", True),
    )
    policy, agent = make_policy(env, agent_cfg, het)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    if agent is not None:
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()
    tiles = [[(int(e), name) for e in het.env_ids_by_task[t][: args_cli.per_task].tolist()]
             for t, name in enumerate(het.task_names)]
    dt = het.step_dt
    steps = int(round(args_cli.seconds / dt)) if args_cli.seconds else het.max_episode_length - 1
    log = dict(scene=args_cli.scene, renderer=args_cli.renderer, checkpoint=args_cli.checkpoint,
               policy=None if args_cli.checkpoint else args_cli.policy, num_envs=het.num_envs, dt=dt, tiles=tiles,
               success=[])
    if rtx:
        aim_cameras(het)
        writer = Writer(Path(args_cli.out), 1.0 / dt)
    else:
        recorder = PoseRecorder(het, PART_SLOTS)
        recorder.record(0.0, [False] * het.num_envs)  # the scene right after the reset
    for step in range(steps):
        with torch.inference_mode():
            if agent is not None:
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if agent is not None and agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0
        success = het._get_curr_successes(het.tp.success_height_threshold, het.tp.success_check_rot).tolist()
        if rtx:
            if bool(dones.any()):
                aim_cameras(het)  # the fixed parts moved at the reset
            frames = het._video_camera.data.output["rgb"].cpu().numpy()
            writer.add(compose(frames, tiles, success, (step + 1) * dt, args_cli.tile, args_cli.per_task))
        else:
            recorder.record((step + 1) * dt, success)
        log["success"].append([[success[e] for e, _ in row] for row in tiles])
    if rtx:
        writer.close()
    else:
        path = recorder.save(args_cli.out, extra=dict(tiles=tiles, scene=args_cli.scene, policy=log["policy"],
                                                      checkpoint=args_cli.checkpoint))
        print(f"[INFO] wrote {path} ({len(recorder.time)} frames); draw it with scripts/render_offline.py {path}")
    Path(args_cli.out).with_suffix(".json").write_text(json.dumps(log), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
