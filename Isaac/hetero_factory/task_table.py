"""Per-task parameter table for the hetero (union-scene) Factory env.

The official Isaac Lab FactoryEnv runs one task per env instance and reads its parameters from scalar
fields of ``cfg.task`` (plus a dozen ``if task.name == ...`` branches). The hetero env runs all three
tasks at once, so every task-dependent value becomes a ``[num_tasks, ...]`` tensor that the env indexes
with each env's task id.

Nothing here is retyped. Each value comes from one of:

* ``cfg``: a field of the official task config (``factory_tasks_cfg.PegInsert`` / ``GearMesh`` / ``NutThread``)
  or env config (``factory_env_cfg.FactoryTask*Cfg``);
* ``fn``: an official function evaluated once per task (``factory_utils.*`` or a ``FactoryEnv`` method
  called on a stand-in ``self`` that only carries ``cfg_task``, ``num_envs`` and ``device``);
* ``replica``: a formula that the official code computes inline inside a method that cannot be called on
  its own; the docstring of each helper names the method it copies.

``FIELD_SOURCES`` records which one applies to each field, and ``to_markdown`` prints the table with it.
Isaac Lab must be importable before calling ``build_task_table`` (launch the app first, see
``print_task_table.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import SimpleNamespace

import torch

TASK_NAMES = ("peg_insert", "gear_mesh", "nut_thread")

# field -> (source kind, where it comes from)
FIELD_SOURCES = {
    "episode_length_s": ("cfg", "FactoryTask*Cfg.episode_length_s"),
    "hand_init_pos": ("cfg", "task.hand_init_pos (relative to the fixed asset tip)"),
    "hand_init_pos_noise": ("cfg", "task.hand_init_pos_noise"),
    "hand_init_orn": ("cfg", "task.hand_init_orn"),
    "hand_init_orn_noise": ("cfg", "task.hand_init_orn_noise"),
    "unidirectional_rot": ("cfg", "task.unidirectional_rot"),
    "fixed_asset_init_pos_noise": ("cfg", "task.fixed_asset_init_pos_noise"),
    "fixed_asset_init_orn_deg": ("cfg", "task.fixed_asset_init_orn_deg"),
    "fixed_asset_init_orn_range_deg": ("cfg", "task.fixed_asset_init_orn_range_deg"),
    "held_asset_pos_noise": ("cfg", "task.held_asset_pos_noise"),
    "held_noise_z_one_sided": ("replica", "FactoryEnv.randomize_initial_state: gear z noise in [-1, 0]"),
    "keypoint_scale": ("cfg", "task.keypoint_scale"),
    "keypoint_coef_baseline": ("cfg", "task.keypoint_coef_baseline"),
    "keypoint_coef_coarse": ("cfg", "task.keypoint_coef_coarse"),
    "keypoint_coef_fine": ("cfg", "task.keypoint_coef_fine"),
    "action_penalty_ee_scale": ("cfg", "task.action_penalty_ee_scale"),
    "action_grad_penalty_scale": ("cfg", "task.action_grad_penalty_scale"),
    "success_height_threshold": ("replica", "FactoryEnv._get_curr_successes with task.success_threshold"),
    "engage_height_threshold": ("replica", "FactoryEnv._get_curr_successes with task.engage_threshold"),
    "success_check_rot": ("replica", "FactoryEnv._get_rewards: check_rot only for nut_thread"),
    "ee_success_yaw": ("cfg", "task.ee_success_yaw"),
    "held_base_pos_local": ("fn", "factory_utils.get_held_base_pos_local"),
    "target_held_base_pos_local": ("fn", "factory_utils.get_target_held_base_pose (identity fixed pose)"),
    "fixed_tip_pos_local": ("replica", "FactoryEnv.randomize_initial_state: fixed_tip_pos_local"),
    "held_in_hand_pos": ("fn", "FactoryEnv.get_handheld_asset_relative_pose"),
    "held_in_hand_quat": ("fn", "FactoryEnv.get_handheld_asset_relative_pose"),
    "gripper_width": ("replica", "FactoryEnv._set_franka_to_default_pose: held diameter / 2 * 1.25"),
    "has_flanking_gears": ("replica", "FactoryEnv: name == gear_mesh and task.add_flanking_gears"),
    "held_friction": ("cfg", "task.held_asset_cfg.friction"),
    "fixed_friction": ("cfg", "task.fixed_asset_cfg.friction"),
    "robot_friction": ("cfg", "task.robot_cfg.friction"),
    "held_mass": ("cfg", "task.held_asset_cfg.mass"),
    "fixed_mass": ("cfg", "task.fixed_asset_cfg.mass"),
}


@dataclass
class TaskTable:
    """One row per task (``names[i]``); every tensor has the task axis first."""

    names: tuple[str, ...]
    num_keypoints: int  # the same for all tasks (checked): keypoint tensors are [num_envs, K, 3]
    episode_length_s: torch.Tensor  # [T]
    hand_init_pos: torch.Tensor  # [T, 3]
    hand_init_pos_noise: torch.Tensor  # [T, 3]
    hand_init_orn: torch.Tensor  # [T, 3]
    hand_init_orn_noise: torch.Tensor  # [T, 3]
    unidirectional_rot: torch.Tensor  # [T] bool
    fixed_asset_init_pos_noise: torch.Tensor  # [T, 3]
    fixed_asset_init_orn_deg: torch.Tensor  # [T]
    fixed_asset_init_orn_range_deg: torch.Tensor  # [T]
    held_asset_pos_noise: torch.Tensor  # [T, 3]
    held_noise_z_one_sided: torch.Tensor  # [T] bool
    keypoint_scale: torch.Tensor  # [T]
    keypoint_coef_baseline: torch.Tensor  # [T, 2]
    keypoint_coef_coarse: torch.Tensor  # [T, 2]
    keypoint_coef_fine: torch.Tensor  # [T, 2]
    action_penalty_ee_scale: torch.Tensor  # [T]
    action_grad_penalty_scale: torch.Tensor  # [T]
    success_height_threshold: torch.Tensor  # [T] metres
    engage_height_threshold: torch.Tensor  # [T] metres
    success_check_rot: torch.Tensor  # [T] bool
    ee_success_yaw: torch.Tensor  # [T]
    held_base_pos_local: torch.Tensor  # [T, 3]
    target_held_base_pos_local: torch.Tensor  # [T, 3]
    fixed_tip_pos_local: torch.Tensor  # [T, 3]
    held_in_hand_pos: torch.Tensor  # [T, 3]
    held_in_hand_quat: torch.Tensor  # [T, 4] (w, x, y, z)
    gripper_width: torch.Tensor  # [T]
    has_flanking_gears: torch.Tensor  # [T] bool
    held_friction: torch.Tensor  # [T]
    fixed_friction: torch.Tensor  # [T]
    robot_friction: torch.Tensor  # [T]
    held_mass: torch.Tensor  # [T]
    fixed_mass: torch.Tensor  # [T]

    def tensor_fields(self) -> list[str]:
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)]

    def per_env(self, task_ids: torch.Tensor) -> SimpleNamespace:
        """Every tensor gathered to ``[num_envs, ...]`` for the given per-env task ids."""
        return SimpleNamespace(**{name: getattr(self, name)[task_ids] for name in self.tensor_fields()})

    def to_markdown(self, decimals: int = 6) -> str:
        """The table as Markdown: one row per field, one column per task, plus where the value comes from."""

        def fmt(x: torch.Tensor) -> str:
            if x.dtype == torch.bool:
                return "yes" if bool(x) else "no"
            vals = x.reshape(-1).tolist()
            out = [f"{v:.{decimals}g}" for v in vals]
            return out[0] if len(out) == 1 else "(" + ", ".join(out) + ")"

        head = "| field | " + " | ".join(self.names) + " | source |"
        rows = [head, "|" + "---|" * (len(self.names) + 2)]
        rows.append("| num_keypoints | " + " | ".join([str(self.num_keypoints)] * len(self.names)) + " | cfg: task.num_keypoints |")
        for name in self.tensor_fields():
            t = getattr(self, name)
            kind, where = FIELD_SOURCES[name]
            rows.append(f"| {name} | " + " | ".join(fmt(t[i]) for i in range(len(self.names))) + f" | {kind}: {where} |")
        return "\n".join(rows)


def load_official():
    """The official task configs and env configs, keyed by task name."""
    from isaaclab_tasks.direct.factory import factory_env_cfg, factory_tasks_cfg

    tasks = {
        "peg_insert": factory_tasks_cfg.PegInsert(),
        "gear_mesh": factory_tasks_cfg.GearMesh(),
        "nut_thread": factory_tasks_cfg.NutThread(),
    }
    env_cfgs = {
        "peg_insert": factory_env_cfg.FactoryTaskPegInsertCfg(),
        "gear_mesh": factory_env_cfg.FactoryTaskGearMeshCfg(),
        "nut_thread": factory_env_cfg.FactoryTaskNutThreadCfg(),
    }
    for name in TASK_NAMES:
        assert tasks[name].name == name, (name, tasks[name].name)
        assert env_cfgs[name].task.name == name, (name, env_cfgs[name].task.name)
    return tasks, env_cfgs


def _height_threshold(task, fraction: float) -> float:
    """Replicates the height threshold inside ``FactoryEnv._get_curr_successes``."""
    fixed_cfg = task.fixed_asset_cfg
    if task.name in ("peg_insert", "gear_mesh"):
        return fixed_cfg.height * fraction
    if task.name == "nut_thread":
        return fixed_cfg.thread_pitch * fraction
    raise NotImplementedError(task.name)


def _fixed_tip_pos_local(task) -> list[float]:
    """Replicates ``fixed_tip_pos_local`` inside ``FactoryEnv.randomize_initial_state``."""
    tip = [0.0, 0.0, task.fixed_asset_cfg.height + task.fixed_asset_cfg.base_height]
    if task.name == "gear_mesh":
        tip[0] = task.fixed_asset_cfg.medium_gear_base_offset[0]
    return tip


def build_task_table(device: str = "cpu", task_names: tuple[str, ...] = TASK_NAMES) -> TaskTable:
    """Evaluate every task-dependent value of the official Factory env once per task."""
    from isaaclab_tasks.direct.factory import factory_utils
    from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

    tasks, env_cfgs = load_official()
    tasks = [tasks[n] for n in task_names]
    env_cfgs = [env_cfgs[n] for n in task_names]

    num_keypoints = {t.num_keypoints for t in tasks}
    assert len(num_keypoints) == 1, f"tasks disagree on num_keypoints: {num_keypoints}"
    robot_friction = {t.robot_cfg.friction for t in tasks}
    assert len(robot_friction) == 1, f"tasks disagree on robot friction (one robot per env): {robot_friction}"

    def f(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def b(values):
        return torch.tensor(values, dtype=torch.bool, device=device)

    held_base, target, in_hand_pos, in_hand_quat = [], [], [], []
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    for t in tasks:
        held_base.append(factory_utils.get_held_base_pos_local(t.name, t.fixed_asset_cfg, 1, device)[0])
        pos, _ = factory_utils.get_target_held_base_pose(
            torch.zeros((1, 3), device=device), identity_quat, t.name, t.fixed_asset_cfg, 1, device
        )
        target.append(pos[0])
        stand_in = SimpleNamespace(cfg_task=t, num_envs=1, device=device)
        rel_pos, rel_quat = FactoryEnv.get_handheld_asset_relative_pose(stand_in)
        in_hand_pos.append(rel_pos[0])
        in_hand_quat.append(rel_quat[0])

    return TaskTable(
        names=tuple(task_names),
        num_keypoints=num_keypoints.pop(),
        episode_length_s=f([c.episode_length_s for c in env_cfgs]),
        hand_init_pos=f([t.hand_init_pos for t in tasks]),
        hand_init_pos_noise=f([t.hand_init_pos_noise for t in tasks]),
        hand_init_orn=f([t.hand_init_orn for t in tasks]),
        hand_init_orn_noise=f([t.hand_init_orn_noise for t in tasks]),
        unidirectional_rot=b([t.unidirectional_rot for t in tasks]),
        fixed_asset_init_pos_noise=f([t.fixed_asset_init_pos_noise for t in tasks]),
        fixed_asset_init_orn_deg=f([t.fixed_asset_init_orn_deg for t in tasks]),
        fixed_asset_init_orn_range_deg=f([t.fixed_asset_init_orn_range_deg for t in tasks]),
        held_asset_pos_noise=f([t.held_asset_pos_noise for t in tasks]),
        held_noise_z_one_sided=b([t.name == "gear_mesh" for t in tasks]),
        keypoint_scale=f([t.keypoint_scale for t in tasks]),
        keypoint_coef_baseline=f([t.keypoint_coef_baseline for t in tasks]),
        keypoint_coef_coarse=f([t.keypoint_coef_coarse for t in tasks]),
        keypoint_coef_fine=f([t.keypoint_coef_fine for t in tasks]),
        action_penalty_ee_scale=f([t.action_penalty_ee_scale for t in tasks]),
        action_grad_penalty_scale=f([t.action_grad_penalty_scale for t in tasks]),
        success_height_threshold=f([_height_threshold(t, t.success_threshold) for t in tasks]),
        engage_height_threshold=f([_height_threshold(t, t.engage_threshold) for t in tasks]),
        success_check_rot=b([t.name == "nut_thread" for t in tasks]),
        ee_success_yaw=f([t.ee_success_yaw for t in tasks]),
        held_base_pos_local=torch.stack(held_base).float(),
        target_held_base_pos_local=torch.stack(target).float(),
        fixed_tip_pos_local=f([_fixed_tip_pos_local(t) for t in tasks]),
        held_in_hand_pos=torch.stack(in_hand_pos).float(),
        held_in_hand_quat=torch.stack(in_hand_quat).float(),
        gripper_width=f([t.held_asset_cfg.diameter / 2 * 1.25 for t in tasks]),
        has_flanking_gears=b([t.name == "gear_mesh" and bool(getattr(t, "add_flanking_gears", False)) for t in tasks]),
        held_friction=f([t.held_asset_cfg.friction for t in tasks]),
        fixed_friction=f([t.fixed_asset_cfg.friction for t in tasks]),
        robot_friction=f([t.robot_cfg.friction for t in tasks]),
        held_mass=f([t.held_asset_cfg.mass for t in tasks]),
        fixed_mass=f([t.fixed_asset_cfg.mass for t in tasks]),
    )
