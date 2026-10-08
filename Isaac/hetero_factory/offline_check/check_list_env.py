"""Offline checks of the big-list scene (HeteroFactoryListEnv) on the fake runtime.

The fake runtime returns part slots in DEScending env order, so any code that assumes "row == env id" fails.

1. Structure: each part slot exists only in the envs of its task; the robot in every env; nothing is parked.
2. Same results as the union scene: with the same layout and seeds, the active parts, observation frame,
   observations and 12 steps of rewards are identical (the union scene's parked parts never enter them).
3. A full episode and the reset after it (the reset with None = all envs); tasks stay fixed.
4. Homo is refused; every rollout holds the fixed split of samples.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "stubs"), str(HERE.parents[1])]

import torch  # noqa: E402

from isaaclab_tasks.direct.factory.factory_env import FactoryEnv  # noqa: E402

from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.hetero_factory_env_cfg import (  # noqa: E402
    PART_SLOTS,
    TASK_SLOTS,
    HeteroFactoryEnvCfg,
    HeteroFactoryListEnvCfg,
)
from hetero_factory.hetero_factory_list_env import HeteroFactoryListEnv  # noqa: E402
from hetero_factory.rl_utils import MetricsObserver  # noqa: E402
from hetero_factory.task_table import TASK_NAMES  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def fake_ik(self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids):
    r = self._robot.data
    i = self.fingertip_body_idx
    r.body_pos_w[env_ids, i] = ctrl_target_fingertip_midpoint_pos[env_ids] + self.scene.env_origins[env_ids]
    r.body_quat_w[env_ids, i] = ctrl_target_fingertip_midpoint_quat[env_ids]
    self.step_sim_no_action()
    return torch.zeros((len(env_ids), 3)), torch.zeros((len(env_ids), 3))


FactoryEnv.set_pos_inverse_kinematics = fake_ik


def make(cls, cfg_cls, n, mode="hetero", layout="interleaved"):
    cfg = cfg_cls()
    cfg.scene = cfg.scene.replace(num_envs=n)
    cfg.assignment = cfg.assignment.__class__()
    cfg.assignment.mode = mode
    cfg.assignment.layout = layout
    return cls(cfg)


N = 7
for layout in ("interleaved", "blocked"):
    torch.manual_seed(5)
    lst = make(HeteroFactoryListEnv, HeteroFactoryListEnvCfg, N, layout=layout)
    ids = lst.task_ids.tolist()
    # ---- 1. structure
    for slot, (task, role, _a, prim) in PART_SLOTS.items():
        t = TASK_NAMES.index(task)
        envs = sorted(e for e in range(N) if ids[e] == t)
        asset = lst._slots[slot]
        check(sorted(asset.env_ids_of_rows) == envs and asset.n == len(envs),
              f"[{layout}] {slot}: {asset.n} instances, only in envs {envs}")
    check(lst._robot.n == N, f"[{layout}] robot in all {N} envs")
    check(lst.cfg.scene.replicate_physics is False, f"[{layout}] envs built independently (replicate_physics=False)")

    # ---- 2. same as the union scene
    torch.manual_seed(5)
    uni = make(HeteroFactoryEnv, HeteroFactoryEnvCfg, N, layout=layout)
    torch.manual_seed(9)
    obs_l, _ = lst.reset()
    torch.manual_seed(9)
    obs_u, _ = uni.reset()
    check(lst.task_ids.tolist() == uni.task_ids.tolist(), f"[{layout}] same task split {lst.task_ids.tolist()}")
    same = lambda a, b: torch.allclose(a, b, atol=1e-6)  # noqa: E731
    check(same(lst.fixed_pos, uni.fixed_pos) and same(lst.fixed_quat, uni.fixed_quat), f"[{layout}] fixed parts = union")
    check(same(lst.held_pos, uni.held_pos) and same(lst.held_quat, uni.held_quat), f"[{layout}] held parts = union")
    check(same(lst.fixed_pos_obs_frame, uni.fixed_pos_obs_frame), f"[{layout}] observation frame = union")
    check(same(obs_l["policy"], obs_u["policy"]) and same(obs_l["critic"], obs_u["critic"]), f"[{layout}] observations = union")
    # each env's active part, read through the row map, is that env's part (rows are reversed in the fake)
    for e in range(N):
        slot = TASK_SLOTS[TASK_NAMES[ids[e]]]["held"]
        asset = lst._slots[slot]
        row = asset.env_ids_of_rows.index(e)
        check(torch.allclose(asset.data.root_pos_w[row] - lst.scene.env_origins[e], lst.held_pos[e]),
              f"[{layout}] env {e}: held part read from row {row} of {slot}")
    check(not any(w for s, a in lst._slots.items() for w in a.writes if any(i >= a.n for i in w[0])),
          f"[{layout}] every write used rows inside the slot")
    gear_envs = [e for e in range(N) if ids[e] == 1]
    for e in gear_envs:
        base = lst._slots["gear_fixed"].data.root_pos_w[lst._slots["gear_fixed"].env_ids_of_rows.index(e)]
        small = lst._slots["gear_small"].data.root_pos_w[lst._slots["gear_small"].env_ids_of_rows.index(e)]
        check(torch.allclose(base, small), f"[{layout}] env {e}: small gear seated on its gear base")
    g = torch.Generator().manual_seed(2)
    rew_same = True
    for step in range(12):
        a = torch.rand((N, 6), generator=g) * 2 - 1
        torch.manual_seed(50 + step)
        _, rl, *_ = lst.step(a.clone())
        torch.manual_seed(50 + step)
        _, ru, *_ = uni.step(a.clone())
        rew_same &= torch.allclose(rl, ru, atol=1e-5)
    check(rew_same, f"[{layout}] rewards over 12 steps = union")

# ---- 3. a full episode and the reset after it
torch.manual_seed(0)
lst = make(HeteroFactoryListEnv, HeteroFactoryListEnvCfg, N)
lst.reset()
before = lst.task_ids.tolist()
for i in range(lst.max_episode_length):
    _, _, _, trunc, extras = lst.step(torch.zeros((N, 6)))
    if trunc.any():
        break
check(bool(trunc.all()) and i + 1 == lst.max_episode_length - 1, f"all envs time out together after {i + 1} steps")
check(lst.task_ids.tolist() == before, f"tasks stay fixed across episodes {before} (leftover env cannot rotate)")
check(all(f"successes_{n}" in extras for n in TASK_NAMES), "per-task success logged")
check(lst.episode_length_buf.sum().item() == 0, "episode counters reset for every env (reset with None = all)")

# ---- 4. homo refused; samples per rollout
try:
    make(HeteroFactoryListEnv, HeteroFactoryListEnvCfg, N, mode="homo")
    check(False, "homo refused")
except ValueError:
    check(True, "homo refused (each env's task is fixed when the scene is built)")
torch.manual_seed(0)
lst = make(HeteroFactoryListEnv, HeteroFactoryListEnvCfg, N)
lst.reset()
with tempfile.TemporaryDirectory() as d:
    obs = MetricsObserver(d, TASK_NAMES, env=lst)
    obs.after_init(SimpleNamespace(epoch_num=0))
    for epoch in (1, 2):
        for _ in range(16):
            _, _, _, _, extras = lst.step(torch.zeros((N, 6)))
            obs.process_infos(extras, None)
        obs.after_print_stats(epoch * 16 * N, epoch, 0.0)
    lines = [json.loads(x) for x in (Path(d) / "metrics.jsonl").read_text().splitlines()]
check(all(ln["samples_per_task"] == {"peg_insert": 48, "gear_mesh": 32, "nut_thread": 32} for ln in lines),
      f"samples per rollout {lines[0]['samples_per_task']}")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
