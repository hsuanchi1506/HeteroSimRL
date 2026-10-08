"""Offline checks of HeteroFactoryEnv on a fake Isaac Lab runtime (no physics; parts move only when written).

1. Equivalence: in homo mode with the same RNG seed, every task must give exactly what the official
   FactoryEnv gives for that task (fixed / held poses, observation frame, observations, rewards, successes).
2. Hetero routing: 7 envs (2 per task + 1 rotating): each env's active parts are placed, the others parked,
   flanking gears only on gear envs, per-env gripper width, nut-only unidirectional rotation, task one-hot.
3. A full 30 s episode: the second reset rotates the leftover env's task; per-task logs exist.
4. Success logic per task, including the nut's rotation condition.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ISAAC = HERE.parents[1]
sys.path[:0] = [str(HERE / "stubs"), str(REPO_ISAAC)]

import torch  # noqa: E402

import isaacsim.core.utils.torch as torch_utils  # noqa: E402
from isaaclab._fake_runtime import STATE  # noqa: E402
from isaaclab_tasks.direct.factory import factory_env_cfg as ecfg  # noqa: E402
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv  # noqa: E402

from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.hetero_factory_env_cfg import PARKING_POS, TASK_SLOTS, HeteroFactoryEnvCfg  # noqa: E402
from hetero_factory.task_table import TASK_NAMES  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


IK_FAIL_ONCE = {"on": False, "done": False}


def fake_ik(self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids):
    """Stand-in for the official reset IK: put the fingertip on the target (optionally fail env 0 once)."""
    r = self._robot.data
    i = self.fingertip_body_idx
    r.body_pos_w[env_ids, i] = ctrl_target_fingertip_midpoint_pos[env_ids] + self.scene.env_origins[env_ids]
    r.body_quat_w[env_ids, i] = ctrl_target_fingertip_midpoint_quat[env_ids]
    self.step_sim_no_action()
    pos_err = torch.zeros((len(env_ids), 3))
    if IK_FAIL_ONCE["on"] and not IK_FAIL_ONCE["done"]:
        pos_err[0, 0] = 0.01
        IK_FAIL_ONCE["done"] = True
    return pos_err, torch.zeros((len(env_ids), 3))


FactoryEnv.set_pos_inverse_kinematics = fake_ik  # inherited by HeteroFactoryEnv


def official_env(task, n):
    cfg = {"peg_insert": ecfg.FactoryTaskPegInsertCfg, "gear_mesh": ecfg.FactoryTaskGearMeshCfg,
           "nut_thread": ecfg.FactoryTaskNutThreadCfg}[task]()
    cfg.scene = cfg.scene.replace(num_envs=n)
    return FactoryEnv(cfg)


def hetero_env(n, mode="hetero", homo_task=0, switch_every_episodes=0, episode_length_s=None):
    cfg = HeteroFactoryEnvCfg()
    cfg.scene = cfg.scene.replace(num_envs=n)
    cfg.assignment = cfg.assignment.__class__()
    cfg.assignment.mode = mode
    cfg.assignment.homo_task = homo_task
    cfg.assignment.homo_switch_every_episodes = switch_every_episodes
    if episode_length_s is not None:
        cfg.episode_length_s = episode_length_s
    return HeteroFactoryEnv(cfg)


def asset_pos(env, asset):
    return asset.data.root_pos_w - env.scene.env_origins


# ------------------------------------------------------------------ 1. equivalence with the official env
N = 5
for k, task in enumerate(TASK_NAMES):
    torch.manual_seed(123)
    off = official_env(task, N)
    torch.manual_seed(7)
    obs_o, _ = off.reset()
    torch.manual_seed(123)
    het = hetero_env(N, "homo", k)
    torch.manual_seed(7)
    obs_h, _ = het.reset()
    slots = TASK_SLOTS[task]
    same = lambda a, b: torch.allclose(a, b, atol=1e-6)  # noqa: E731
    check(same(asset_pos(off, off._fixed_asset), asset_pos(het, het._slots[slots["fixed"]])),
          f"[{task}] homo: fixed part pose = official")
    check(same(off._fixed_asset.data.root_quat_w, het._slots[slots["fixed"]].data.root_quat_w),
          f"[{task}] homo: fixed part orientation = official")
    check(same(asset_pos(off, off._held_asset), asset_pos(het, het._slots[slots["held"]])),
          f"[{task}] homo: held part pose = official")
    check(same(off._held_asset.data.root_quat_w, het._slots[slots["held"]].data.root_quat_w),
          f"[{task}] homo: held part orientation = official")
    check(same(off.fixed_pos_obs_frame, het.fixed_pos_obs_frame), f"[{task}] homo: observation frame = official")
    if task == "gear_mesh":
        for role, oa in (("small_gear", off._small_gear_asset), ("large_gear", off._large_gear_asset)):
            check(same(asset_pos(off, oa), asset_pos(het, het._slots[slots[role]])),
                  f"[{task}] homo: {role} pose = official")
    po, ph = obs_o["policy"], obs_h["policy"]
    # hetero policy input = official 13 + one-hot 3 + prev_actions 6
    check(same(po[:, :13], ph[:, :13]) and same(po[:, 13:], ph[:, 16:]), f"[{task}] homo: policy input = official + one-hot")
    check(torch.all(ph[:, 13:16] == torch.nn.functional.one_hot(torch.full((N,), k), 3)).item(),
          f"[{task}] homo: one-hot marks task {k}")
    so, sh = obs_o["critic"], obs_h["critic"]
    check(same(so[:, :37], sh[:, :37]) and same(so[:, 37:], sh[:, 40:]), f"[{task}] homo: critic input = official + one-hot")
    g = torch.Generator().manual_seed(1)
    rew_same = succ_same = True
    for step in range(12):
        a = torch.rand((N, 6), generator=g) * 2 - 1
        torch.manual_seed(100 + step)
        _, ro, *_ = off.step(a.clone())
        torch.manual_seed(100 + step)
        _, rh, *_ = het.step(a.clone())
        rew_same &= torch.allclose(ro, rh, atol=1e-5)
        succ_same &= torch.allclose(off.actions, het.actions, atol=1e-6)
    check(rew_same, f"[{task}] homo: rewards over 12 steps = official")
    check(succ_same, f"[{task}] homo: actions after the decimation loop = official (incl. nut in-place rewrite)")
    if task == "nut_thread":
        nut_episode_steps = off.max_episode_length

# ------------------------------------------------------------------ 2. hetero routing
torch.manual_seed(0)
IK_FAIL_ONCE.update(on=True, done=False)
env = hetero_env(7, "hetero")
obs, _ = env.reset()
IK_FAIL_ONCE["on"] = False
ids = env.task_ids.tolist()
check(ids == [0, 1, 2, 0, 1, 2, 0], f"episode 0 tasks {ids}")
check(env.last_reset_ik_attempts == 1, "IK retry counted (env 0 failed once)")
check(STATE["sim"].gravity_calls[-2:] == [(0.0, 0.0, 0.0), (0.0, 0.0, -9.81)], "gravity off during reset, restored after")
origins = env.scene.env_origins
for slot, (task, role, _a, _p) in __import__("hetero_factory.hetero_factory_env_cfg", fromlist=["PART_SLOTS"]).PART_SLOTS.items():
    t = TASK_NAMES.index(task)
    pos = env._slots[slot].data.root_pos_w - origins
    for e in range(7):
        parked = torch.allclose(pos[e], torch.tensor(PARKING_POS[slot]))
        if ids[e] == t:
            near_table = pos[e, 0].item() > 0.3
            check(near_table and not parked, f"env {e} ({task}): {slot} on the table at x={pos[e, 0]:.3f}")
        else:
            check(parked, f"env {e} ({TASK_NAMES[ids[e]]}): {slot} parked")
gear_envs = [e for e in range(7) if ids[e] == 1]
for e in gear_envs:
    base = env._slots["gear_fixed"].data.root_pos_w[e]
    check(torch.allclose(env._slots["gear_small"].data.root_pos_w[e], base)
          and torch.allclose(env._slots["gear_large"].data.root_pos_w[e], base), f"env {e}: flanking gears on the gear base")
widths = env._robot.data.joint_pos[:, 7]
check(torch.allclose(widths, env.task_table.gripper_width[env.task_ids]), f"gripper opened to each task's width {[round(w, 5) for w in widths.tolist()]}")
check(obs["policy"].shape == (7, 22) and obs["critic"].shape == (7, 46), f"input shapes {tuple(obs['policy'].shape)} {tuple(obs['critic'].shape)}")
check(torch.all(obs["policy"][:, 13:16].argmax(-1) == env.task_ids).item(), "one-hot = task per env")
tip_expect = env.fixed_pos + torch_utils.quat_apply(env.fixed_quat, env.task_table.fixed_tip_pos_local[env.task_ids])
check(torch.allclose(env.fixed_pos_obs_frame, tip_expect, atol=1e-6), "observation frame = fixed pose + that task's tip offset")
# held part sits in the hand: within a few cm of the fingertip
d = torch.linalg.norm(env.held_pos - env.fingertip_midpoint_pos, dim=-1)
check(bool((d < 0.06).all()), f"held parts in the hand (fingertip distance {[round(x, 3) for x in d.tolist()]})")
# unidirectional rotation only on nut envs
env.actions[:] = 0.5
env._apply_action()
nut = env.task_ids == 2
check(torch.allclose(env.actions[nut, 5], torch.tensor(-0.75)) and torch.allclose(env.actions[~nut, 5], torch.tensor(0.5)),
      "rotation remap only on nut envs (in place, like the official code)")

# ------------------------------------------------------------------ 3. one full episode, then the next reset
env.actions[:] = 0
steps = env.max_episode_length
check(steps == nut_episode_steps, f"episode length {steps} = the official nut_thread episode ({nut_episode_steps})")
got_success_logs = False
for i in range(steps):
    _, r, term, trunc, extras = env.step(torch.zeros((7, 6)))
    if trunc.any():
        got_success_logs = all(f"successes_{n}" in extras for n in TASK_NAMES)
        break
check(i + 1 == steps - 1 and bool(trunc.all()), f"all envs time out together after {i + 1} steps")
check(got_success_logs, "per-task success logged at the episode end")
check(all(f"rew_{n}" in env.extras for n in TASK_NAMES), "per-task reward logged every step")
ids2 = env.task_ids.tolist()
check(ids2 == [0, 1, 2, 0, 1, 2, 1], f"episode 1 tasks {ids2}: the leftover env rotated 0 -> 1")
check(torch.allclose(env._slots["peg_held"].data.root_pos_w[6] - origins[6], torch.tensor(PARKING_POS["peg_held"])),
      "env 6 switched peg -> gear: its peg is parked again")
check(not torch.allclose(env._slots["gear_held"].data.root_pos_w[6] - origins[6], torch.tensor(PARKING_POS["gear_held"])),
      "env 6 switched peg -> gear: its medium gear is in the hand")

# ------------------------------------------------------------------ 4. success logic per task
env._compute_intermediate_values(env.physics_dt)
tp = env.tp
# put every held base exactly on its target
target_pos, _ = env._target_held_base_pose()
held_base_now, _ = env._held_base_pose()
env.held_pos = env.held_pos + (target_pos - held_base_now)
for name, yaw in (("hand yaw 300 deg", 300.0), ("hand yaw 90 deg", 90.0)):
    q = torch_utils.quat_from_euler_xyz(torch.full((7,), math.pi), torch.zeros(7), torch.full((7,), math.radians(yaw)))
    env.fingertip_midpoint_quat = q
    s = env._get_curr_successes(tp.success_height_threshold, tp.success_check_rot)
    expect = [True] * 7 if yaw == 300.0 else [t != 2 for t in env.task_ids.tolist()]
    check(s.tolist() == expect, f"success with held parts on target, {name}: {s.int().tolist()}")
env.held_pos[:, 2] += 0.0009  # 0.9 mm above target: > nut (0.75 mm), < peg / gear (1 mm)
s = env._get_curr_successes(tp.success_height_threshold, torch.zeros(7, dtype=torch.bool))
check(s.tolist() == [t != 2 for t in env.task_ids.tolist()], f"0.9 mm high: peg/gear succeed, nut does not {s.int().tolist()}")

# ------------------------------------------------------------------ 5. homo switching, driven like rl_games
import json  # noqa: E402
import tempfile  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from hetero_factory.rl_utils import MetricsObserver, make_homo_rollout_hook  # noqa: E402


def rl_games_like(env, epochs, horizon, hook):
    """rl_games' rhythm: `horizon` env steps per iteration, then after_print_stats (twice, as in the fork)."""
    with tempfile.TemporaryDirectory() as d:
        obs = MetricsObserver(d, TASK_NAMES, env=env, epoch_end_hook=hook)
        obs.after_init(SimpleNamespace(epoch_num=0))
        frame = 0
        for epoch in range(1, epochs + 1):
            for _ in range(horizon):
                _, _, _, _, extras = env.step(torch.zeros((env.num_envs, 6)))
                obs.process_infos(extras, None)
            frame += horizon * env.num_envs
            obs.after_print_stats(frame, epoch, 0.0)
            obs.after_print_stats(frame, epoch, 0.0)
        lines = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
        eps_path = Path(d) / "episodes.jsonl"
        eps = [json.loads(x) for x in eps_path.read_text().splitlines()] if eps_path.exists() else []
    return lines, eps


H = 16
for block in (1, 2):
    torch.manual_seed(0)
    env = hetero_env(4, "homo", 0)
    env.reset()
    lines, eps = rl_games_like(env, 3 * block + 1, H, make_homo_rollout_hook(env, block, H))
    per_epoch = [[t for t, c in ln["samples_per_task"].items() if c] for ln in lines]
    expect = [[TASK_NAMES[(e // block) % 3]] for e in range(3 * block + 1)]
    check(per_epoch == expect, f"homo rollout switch, block {block}: one task per rollout {[p[0][:3] for p in per_epoch]}")
    check(all(sum(ln["samples_per_task"].values()) == 4 * H for ln in lines), f"block {block}: {4 * H} samples per rollout")
    n_switch = (3 * block + 1) // block  # a switch at the end of every block-th iteration
    check(len(eps) == n_switch and all(e["truncated"] and e["steps"] == block * H for e in eps),
          f"block {block}: {n_switch} cut episodes, each logged as truncated after {block * H} steps")
    check([e["task"] for e in eps] == [TASK_NAMES[i % 3] for i in range(n_switch)],
          f"block {block}: truncated episodes cycle peg, gear, nut {[e['task'][:3] for e in eps]}")
check("successes_peg_insert_truncated" in env.extras and "return_nut_thread_truncated" in env.extras,
      "truncated episodes use their own keys")

# a natural time-out in the middle of a block keeps the task; the switch still lands on the block's last step
torch.manual_seed(0)
env = hetero_env(4, "homo", 0, episode_length_s=2.0)  # 30 steps -> episodes end after 29 steps
env.reset()
lines, eps = rl_games_like(env, 6, H, make_homo_rollout_hook(env, 3, H))
per_epoch = [[t for t, c in ln["samples_per_task"].items() if c] for ln in lines]
check(per_epoch == [["peg_insert"]] * 3 + [["gear_mesh"]] * 3, "natural episode ends inside a block keep the task")
got = [(e["task"], e["steps"], e["truncated"]) for e in eps][:3]
check(got == [("peg_insert", 29, False), ("peg_insert", 19, True), ("gear_mesh", 29, False)],
      f"episodes: a full 29-step episode, then the one cut at the switch after 19 steps {got}")

# episode switching: every complete episode, no truncation
torch.manual_seed(0)
env = hetero_env(4, "homo", 0, switch_every_episodes=1, episode_length_s=2.0)
env.reset()
lines, eps = rl_games_like(env, 6, H, None)
check([e["task"] for e in eps] == ["peg_insert", "gear_mesh", "nut_thread"] and not any(e["truncated"] for e in eps),
      f"homo episode switch: {[e['task'][:3] for e in eps]}, none truncated")
mixed = sum(1 for ln in lines if sum(1 for c in ln["samples_per_task"].values() if c) > 1)
check(mixed > 0, f"episode switching: {mixed} of {len(lines)} rollouts hold two tasks (the trade-off)")

# hetero: every rollout holds all three tasks, in the same proportions
torch.manual_seed(0)
env = hetero_env(7, "hetero")
env.reset()
lines, _ = rl_games_like(env, 2, H, None)
check(all(ln["samples_per_task"] == {"peg_insert": 3 * H, "gear_mesh": 2 * H, "nut_thread": 2 * H} for ln in lines),
      f"hetero: samples per rollout {lines[0]['samples_per_task']}")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
