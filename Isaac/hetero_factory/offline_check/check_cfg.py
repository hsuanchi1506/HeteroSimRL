"""Offline check of hetero_factory (task table, assignment, union-scene cfg) against the official Factory
source files, with Isaac Lab replaced by minimal stubs. Writes TASK_PARAMS.md into the repo."""
import math
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ISAAC = HERE.parents[1]
sys.path[:0] = [str(HERE / "stubs"), str(REPO_ISAAC)]

import torch  # noqa: E402

from hetero_factory import assignment as asg  # noqa: E402
from hetero_factory.task_table import TASK_NAMES, build_task_table  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def close(a, b, tol=1e-6):
    return torch.allclose(torch.as_tensor(a, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32), atol=tol)


# ---------------------------------------------------------------- task table
tt = build_task_table("cpu")
P, G, N = (TASK_NAMES.index(n) for n in ("peg_insert", "gear_mesh", "nut_thread"))
# expected values worked out by hand from factory_tasks_cfg.py / factory_env.py / factory_utils.py
check(tt.num_keypoints == 4, "num_keypoints 4 for all tasks")
check(close(tt.episode_length_s, [10.0, 20.0, 30.0]), "episode lengths 10 / 20 / 30 s")
check(close(tt.success_height_threshold, [0.025 * 0.04, 0.02 * 0.05, 0.002 * 0.375]), "success height thresholds")
check(close(tt.engage_height_threshold, [0.025 * 0.9, 0.02 * 0.9, 0.002 * 0.5]), "engage height thresholds")
check(tt.success_check_rot.tolist() == [False, False, True], "rotation check only for the nut")
check(tt.unidirectional_rot.tolist() == [False, False, True], "unidirectional rotation only for the nut")
check(close(tt.held_base_pos_local, [[0, 0, 0], [0.02025, 0, 0], [0, 0, 0.01]]), "held base offsets")
check(close(tt.target_held_base_pos_local, [[0, 0, 0], [0.02025, 0, 0], [0, 0, 0.01 + 0.025 - 0.003]]),
      "target offsets (nut: head + shank - 1.5 pitch)")
check(close(tt.fixed_tip_pos_local, [[0, 0, 0.025], [0.02025, 0, 0.025], [0, 0, 0.035]]), "fixed tip offsets")
check(close(tt.held_in_hand_pos, [[0, 0, 0.05 - 0.017608], [0.02025, 0, 0.03 / 2 * 1.1], [0, 0, 0.01]]),
      "held-in-hand positions")
s = math.sqrt(0.5)
check(close(tt.held_in_hand_quat, [[1, 0, 0, 0], [1, 0, 0, 0], [s, 0, 0, -s]], 1e-5),
      "held-in-hand quats (nut rotated -90 deg about z; gear's -90 is unused by the official code)")
check(close(tt.gripper_width, [0.007986 / 2 * 1.25, 0.03 / 2 * 1.25, 0.024 / 2 * 1.25]), "gripper widths")
check(tt.has_flanking_gears.tolist() == [False, True, False], "flanking gears only for gear_mesh")
check(tt.held_noise_z_one_sided.tolist() == [False, True, False], "one-sided z noise only for gear_mesh")
check(close(tt.keypoint_scale, [0.15, 0.15, 0.05]), "keypoint scales")
check(close(tt.keypoint_coef_fine, [[100, 0], [100, 0], [1500, 0]]), "fine keypoint coefficients")
check(close(tt.held_friction, [0.75, 0.75, 0.01]), "held frictions (nut 0.01)")
check(close(tt.fixed_asset_init_orn_range_deg, [360, 15, 30]), "fixed asset yaw ranges")

ids = torch.tensor([2, 0, 1, 2])
pe = tt.per_env(ids)
check(pe.success_height_threshold.shape == (4,) and close(pe.success_height_threshold[0], 0.00075),
      "per_env gathers by task id")
check(pe.hand_init_pos.shape == (4, 3), "per_env keeps trailing dims")

# ---------------------------------------------------------------- assignment
check(asg.hetero_assignment(3, 3).tolist() == [0, 1, 2], "3 envs: one per task")
for layout in ("interleaved", "blocked"):
    for n in (128, 4096):
        totals = [0, 0, 0]
        for ep in range(3):
            a = asg.hetero_assignment(n, 3, ep, layout)
            c = asg.task_counts(a, 3)
            check(max(c) - min(c) <= 1, f"{layout} {n} envs ep {ep}: counts {c} differ by at most 1")
            totals = [t + x for t, x in zip(totals, c)]
        check(len(set(totals)) == 1, f"{layout} {n} envs: equal env-episodes per task over 3 episodes {totals}")
check(asg.hetero_assignment(6, 3, layout="interleaved").tolist() == [0, 1, 2, 0, 1, 2], "interleaved order")
check(asg.hetero_assignment(6, 3, layout="blocked").tolist() == [0, 0, 1, 1, 2, 2], "blocked order")
check(asg.homo_assignment(5, 2).tolist() == [2] * 5, "homo: all envs one task")

# ---------------------------------------------------------------- union-scene cfg
from hetero_factory import hetero_factory_env_cfg as hc  # noqa: E402
from isaaclab_tasks.direct.factory import factory_tasks_cfg as tcfg  # noqa: E402

cfg = hc.HeteroFactoryEnvCfg()
obs = sum(hc.HETERO_OBS_DIM_CFG[o] for o in cfg.obs_order) + cfg.action_space
state = sum(hc.HETERO_STATE_DIM_CFG[o] for o in cfg.state_order) + cfg.action_space
check((obs, state) == (22, 46), f"policy / critic input dims {obs} / {state} (official 19 / 43 + 3)")
prims = [getattr(cfg, s).prim_path for s in hc.PART_SLOTS]
check(len(set(prims)) == len(prims), "8 slots, distinct prim paths")
official = {"peg_insert": tcfg.PegInsert(), "gear_mesh": tcfg.GearMesh(), "nut_thread": tcfg.NutThread()}
for slot, (task, role, attr, prim) in hc.PART_SLOTS.items():
    ours, theirs = getattr(cfg, slot), getattr(official[task], attr)
    check(ours.spawn.usd_path == theirs.spawn.usd_path and vars(ours.init_state) == vars(theirs.init_state)
          and ours.prim_path == f"/World/envs/env_.*/{prim}"
          and vars(ours.spawn.mass_props) == vars(theirs.spawn.mass_props),
          f"slot {slot}: official usd/mass/init state, prim {prim}")
check(theirs.prim_path == "/World/envs/env_.*/HeldAsset", "official cfg objects left unchanged by replace()")
check(sorted(hc.TASK_SLOTS) == sorted(TASK_NAMES) and set(hc.TASK_SLOTS["gear_mesh"]) ==
      {"fixed", "held", "small_gear", "large_gear"}, "task -> slot map")
stack = {"gear_fixed", "gear_small", "gear_large"}
for a, b in combinations(hc.PARKING_POS, 2):
    if {a, b} <= stack:
        continue
    d = math.dist(hc.PARKING_POS[a][:2], hc.PARKING_POS[b][:2])
    check(d >= 0.25, f"parking {a} - {b}: {d:.2f} m apart")
check(all(abs(x) < 0.95 and abs(y) < 0.95 for x, y, _ in hc.PARKING_POS.values()), "parking inside the 2 m env cell")
check(all(p[0] <= -0.5 for p in hc.PARKING_POS.values()), "parking behind the robot (x <= -0.5)")
check(cfg.episode_length_s == 30.0 and cfg.episode_length_mode == "max", "episode length: max of the tasks (open question)")

# ---------------------------------------------------------------- write the table
md = REPO_ISAAC / "hetero_factory" / "TASK_PARAMS.md"
header = """# Factory 每個任務的參數表

由 `task_table.py` 產生：每個值都來自 Isaac Lab 官方的 Factory 設定或函式（`source` 欄寫出處）。
- 產生方式：2026-10-04 在沒有 Isaac Lab 的筆電上，用最小的替身模組載入官方原始檔（IsaacLab `main` 的
  `direct/factory/`，和 v2.3.2 完全一樣）算出來的。
- 在訓練機器上要重算一次並比對：`./isaaclab.sh -p Isaac/hetero_factory/print_task_table.py --headless`。
- 單位：長度公尺、角度依官方欄位（`*_deg` 是度，`hand_init_orn` 是弧度）、四元數是 (w, x, y, z)。

"""
md.write_text(header + tt.to_markdown() + "\n", encoding="utf-8")
print(f"\nwrote {md}")
print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
