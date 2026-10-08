"""HeteroFactoryEnv: the official Isaac Lab FactoryEnv with the three Factory tasks in one batch.

Subclass of the official ``FactoryEnv`` (``isaaclab_tasks/direct/factory/factory_env.py``). Methods that do
not depend on the task are inherited unchanged: the controller (``generate_ctrl_signals``,
``close_gripper_in_place``), the reset IK (``set_pos_inverse_kinematics``), ``step_sim_no_action``,
``_pre_physics_step``, ``_get_observations``, ``_get_dones``. Every override below is a copy of the
official method in which the single task (``self.cfg_task.<field>`` and the ``if task name == ...`` branches)
is replaced by a per-env value from ``self.tp`` (the task table of ``task_table.py`` gathered by each env's
task id). Each change is marked ``HETERO``; everything else in an override is the official code, in the same
order, including its quirks (noted where they matter).

Scene: the union scene of ``hetero_factory_env_cfg.py``. Each env runs one task per episode; the parts of the
other tasks are parked behind the robot. As in the official env, all envs reset together; each env's task
for the next episode is chosen at the start of every reset (``assignment.py``).

Not run yet: written 2026-10-04 before access to the GPU machine, checked only offline with mocks.
"""

from __future__ import annotations

import re

import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

from .assignment import counts_assignment, hetero_assignment, homo_assignment
from .hetero_factory_env_cfg import (
    HETERO_OBS_DIM_CFG,
    HETERO_STATE_DIM_CFG,
    PART_SLOTS,
    TASK_SLOTS,
    HeteroFactoryEnvCfg,
)
from .task_table import build_task_table


class HeteroFactoryEnv(FactoryEnv):
    cfg: HeteroFactoryEnvCfg

    def __init__(self, cfg: HeteroFactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # HETERO: input sizes include the task one-hot (official: OBS_DIM_CFG / STATE_DIM_CFG)
        cfg.observation_space = sum([HETERO_OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([HETERO_STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        if cfg.episode_length_mode != "max":
            raise NotImplementedError(f"episode_length_mode={cfg.episode_length_mode!r} (team decision pending)")
        # HETERO: no single task. Any official method that still reads cfg_task fails loudly.
        self.cfg_task = None
        self.task_names = tuple(cfg.task_names)
        self.num_tasks = len(self.task_names)

        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self.task_table = build_task_table(self.device, self.task_names)
        self._init_tensors()
        self._set_default_dynamics_parameters()

    # ------------------------------------------------------------------ tasks per env

    def set_homo_task(self, task: int | str):
        """Homo mode: the task every env runs from the next reset on (called by the training loop)."""
        self.homo_task = self.task_names.index(task) if isinstance(task, str) else int(task)

    def next_homo_task(self) -> int:
        """The task after the current one in the cycle peg -> gear -> nut -> peg (RULES.md section 4.A)."""
        return (self.homo_task + 1) % self.num_tasks

    def schedule_homo_switch(self, after_steps: int, task: int | None = None):
        """Homo mode, switching at a rollout boundary: after ``after_steps`` more env steps every env times out
        (a truncation, so PPO bootstraps the value) and the reset that follows starts ``task`` (default: the
        next one). The training loop arms this so the forced reset lands on the last step of a rollout, which
        keeps every rollout single-task."""
        self._switch_countdown = int(after_steps)
        self._switch_task = self.next_homo_task() if task is None else int(task)

    def _get_dones(self):
        """Check which environments are terminated (official), plus the scheduled homo switch."""
        terminated, time_out = super()._get_dones()
        # HETERO: samples collected per task (this step's task ids, before any reset), drained per PPO epoch
        self._samples_per_task += torch.bincount(self.task_ids, minlength=self.num_tasks)
        # HETERO: a scheduled homo switch forces every env to time out on this step
        self._forced_reset = False
        if self._switch_countdown is not None:
            self._switch_countdown -= 1
            if self._switch_countdown <= 0:
                time_out = torch.ones_like(time_out)
                terminated = time_out  # the official env returns the time-out tensor for both
                self._forced_reset = True
                self.homo_task = self._switch_task
                self._switch_countdown = None
        return terminated, time_out

    def pop_samples_per_task(self) -> list[int]:
        """Env steps collected per task since the last call."""
        out = self._samples_per_task.tolist()
        self._samples_per_task.zero_()
        return out

    def _next_task_ids(self) -> torch.Tensor:
        a = self.cfg.assignment
        if a.mode == "hetero" and getattr(a, "counts", None):
            return counts_assignment(a.counts, self.num_envs, self.device)
        if a.mode == "hetero":
            return hetero_assignment(self.num_envs, self.num_tasks, self.episode_count, a.layout, self.device)
        if a.mode == "homo":
            return homo_assignment(self.num_envs, self.homo_task, self.device)
        raise ValueError(f"unknown assignment mode {a.mode!r}")

    def _set_task_ids(self, task_ids: torch.Tensor):
        """Store each env's task and gather its parameters (``self.tp``) from the task table."""
        self.task_ids = task_ids.to(self.device).long()
        self.tp = self.task_table.per_env(self.task_ids)
        self.env_ids_by_task = [(self.task_ids == t).nonzero(as_tuple=False).squeeze(-1) for t in range(self.num_tasks)]
        self.task_onehot = torch.nn.functional.one_hot(self.task_ids, self.num_tasks).float()
        self._update_role_rows()

    # Slot access. In the union scene every env has every slot, so a global env id is also the row of that env
    # in the slot's Articulation. The big-list scene (hetero_factory_list_env.py) overrides _slot_rows and
    # _slot_count, because there a slot exists only in the envs of its task.
    def _slot_rows(self, slot: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Rows of ``env_ids`` in the slot's Articulation."""
        return env_ids

    def _slot_count(self, slot: str) -> int:
        """Number of instances of the slot's Articulation."""
        return self.num_envs

    def _slot_pose(self, slot: str, env_ids: torch.Tensor):
        data = self._slots[slot].data
        rows = self._slot_rows(slot, env_ids)
        return data.root_pos_w[rows], data.root_quat_w[rows]

    def _slot_default_state(self, slot: str, env_ids: torch.Tensor) -> torch.Tensor:
        return self._slots[slot].data.default_root_state.clone()[self._slot_rows(slot, env_ids)]

    def _slot_write(self, slot: str, state: torch.Tensor, env_ids: torch.Tensor):
        """Write ``state`` ([len(env_ids), 13], world frame) to the slot's part in ``env_ids``."""
        if len(env_ids):
            asset = self._slots[slot]
            rows = self._slot_rows(slot, env_ids)
            asset.write_root_pose_to_sim(state[:, 0:7], env_ids=rows)
            asset.write_root_velocity_to_sim(state[:, 7:], env_ids=rows)
            asset.reset()

    def _active_root_pose(self, role: str):
        """World pose of each env's active part with this role ("fixed" or "held"), quaternion (w, x, y, z)."""
        if getattr(self.cfg, "fast_active_pose", False) and getattr(self, "_role_rows", None) is not None:
            # one PhysX read over all tasks' slots of this role, then each env picks its task's row
            pose = self._role_views[role].get_root_transforms()[self._role_rows[role]]
            return pose[:, 0:3], math_utils.convert_quat(pose[:, 3:7], to="wxyz")
        return self._active_root_pose_by_task(role)

    def _active_root_pose_by_task(self, role: str):
        """Same values as ``_active_root_pose``, read one task at a time (one PhysX read per task)."""
        pos = torch.zeros((self.num_envs, 3), device=self.device)
        quat = torch.zeros((self.num_envs, 4), device=self.device)
        for t, ids in enumerate(self.env_ids_by_task):
            if ids.numel():
                pos[ids], quat[ids] = self._slot_pose(TASK_SLOTS[self.task_names[t]][role], ids)
        return pos, quat

    def _build_role_views(self):
        """HETERO: one PhysX articulation view per role over the three tasks' slots (in the union scene 3 x num_envs
        rows, in the big-list scene only the envs that hold each slot), and a table (task, env) -> row."""
        sim_view = self._robot._physics_sim_view
        self._role_views, self._role_table = {}, {}
        for role in ("fixed", "held"):
            paths, key = [], {}
            for t, name in enumerate(self.task_names):
                for p in self._slots[TASK_SLOTS[name][role]].root_physx_view.prim_paths:
                    paths.append(p)
                    key[p] = (t, int(re.search(r"/env_(\d+)(/|$)", p).group(1)))
            view = sim_view.create_articulation_view(paths)
            table = torch.full((self.num_tasks, self.num_envs), -1, dtype=torch.long, device=self.device)
            for row, p in enumerate(view.prim_paths):
                t, e = key[p]
                table[t, e] = row
            self._role_views[role] = view
            self._role_table[role] = table
        self._update_role_rows()

    def _update_role_rows(self):
        """Row of each env's active part in the role views (called whenever the tasks change)."""
        if getattr(self, "_role_table", None) is None:
            return
        env = torch.arange(self.num_envs, device=self.device)
        self._role_rows = {role: table[self.task_ids, env] for role, table in self._role_table.items()}
        if any(bool((rows < 0).any()) for rows in self._role_rows.values()):
            raise RuntimeError("an env's active part is missing from the role views")

    def _active_default_root_state(self, role: str, env_ids: torch.Tensor) -> torch.Tensor:
        """``default_root_state`` (the official spawn pose) of each env's active part, for ``env_ids``."""
        state = torch.zeros((len(env_ids), 13), device=self.device)
        tids = self.task_ids[env_ids]
        for t in range(self.num_tasks):
            mask = tids == t
            if mask.any():
                state[mask] = self._slot_default_state(TASK_SLOTS[self.task_names[t]][role], env_ids[mask])
        return state

    def _write_active_root_state(self, role: str, state: torch.Tensor, env_ids: torch.Tensor):
        """Write ``state`` ([len(env_ids), 13], world frame) to each env's active part with this role."""
        tids = self.task_ids[env_ids]
        for t in range(self.num_tasks):
            mask = tids == t
            if mask.any():
                self._slot_write(TASK_SLOTS[self.task_names[t]][role], state[mask], env_ids[mask])

    def _park(self, slot: str, env_ids: torch.Tensor):
        if len(env_ids):
            self._slot_write(slot, self._parking_state[slot][env_ids], env_ids)

    def _held_base_pose(self):
        """Official ``factory_utils.get_held_base_pose`` with each env's own offset."""
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        held_base_quat, held_base_pos = torch_utils.tf_combine(
            self.held_quat, self.held_pos, identity, self.tp.held_base_pos_local
        )
        return held_base_pos, held_base_quat

    def _target_held_base_pose(self):
        """Official ``factory_utils.get_target_held_base_pose`` with each env's own offset."""
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        target_quat, target_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, identity, self.tp.target_held_base_pos_local
        )
        return target_pos, target_quat

    # ------------------------------------------------------------------ setup

    def _init_tensors(self):
        super()._init_tensors()
        # HETERO: per-env tasks and parking poses
        self.episode_count = 0
        self.homo_task = int(self.cfg.assignment.homo_task)
        self.last_reset_ik_attempts = 0
        self._ep_return = torch.zeros((self.num_envs,), device=self.device)
        self._samples_per_task = torch.zeros((self.num_tasks,), dtype=torch.long, device=self.device)
        self._switch_countdown = None  # homo switch at a rollout boundary (schedule_homo_switch)
        self._switch_task = None
        self._forced_reset = False
        self._last_success_by_task = {}
        # one record per task per finished episode, drained by the training script's observer
        self.episode_log = []
        self._set_task_ids(self._next_task_ids())
        self._parking_state = {}
        for slot in PART_SLOTS:
            state = torch.zeros((self.num_envs, 13), device=self.device)
            state[:, 0:3] = torch.tensor(self.cfg.parking_pos[slot], device=self.device) + self.scene.env_origins
            state[:, 3] = 1.0
            self._parking_state[slot] = state
        if getattr(self.cfg, "fast_active_pose", False):
            if hasattr(getattr(self._robot, "_physics_sim_view", None), "create_articulation_view"):
                self._build_role_views()
            else:  # offline checks (stubbed simulator): read one task at a time
                print("[INFO] no PhysX simulation view: reading active parts one task at a time")

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        # HETERO: each held / fixed part gets its own task's friction. The small and large gears are not set,
        # as in the official env.
        table = self.task_table
        for slot, (task, role, _attr, _prim) in PART_SLOTS.items():
            if slot not in self._slots:
                continue
            t = self.task_names.index(task)
            if role == "held":
                factory_utils.set_friction(self._slots[slot], float(table.held_friction[t]), self._slot_count(slot))
            elif role == "fixed":
                factory_utils.set_friction(self._slots[slot], float(table.fixed_friction[t]), self._slot_count(slot))
        factory_utils.set_friction(self._robot, float(table.robot_friction[0]), self.scene.num_envs)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        # HETERO: one Articulation per part slot (official: fixed + held asset, plus two gears for gear_mesh)
        self._slots = {slot: Articulation(getattr(self.cfg, slot)) for slot in PART_SLOTS}
        # HETERO: extra prims (e.g. the video script's cameras) go in before cloning, like the official camera tasks
        self._setup_extra_scene()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        for slot, asset in self._slots.items():
            self.scene.articulations[slot] = asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ per step

    def _setup_extra_scene(self):
        """Hook for prims added to every env before cloning (nothing here; the video script adds cameras)."""

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # HETERO: fixed / held pose from each env's active part (official: self._fixed_asset / self._held_asset)
        fixed_pos_w, self.fixed_quat = self._active_root_pose("fixed")
        held_pos_w, self.held_quat = self._active_root_pose("held")
        self.fixed_pos = fixed_pos_w - self.scene.env_origins
        self.held_pos = held_pos_w - self.scene.env_origins

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        obs_dict, state_dict = super()._get_factory_obs_state_dict()
        # HETERO: which task this env runs (RULES.md: shared policy with a task id)
        obs_dict["task_onehot"] = self.task_onehot
        state_dict["task_onehot"] = self.task_onehot
        return obs_dict, state_dict

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        # HETERO: per env (official: if cfg_task.unidirectional_rot, for every env). As in the official code,
        # rot_actions is a view of self.actions, so this rewrites self.actions in place, once per physics step
        # of the decimation loop.
        # torch.where instead of a mask test keeps the GPU from syncing with the CPU every physics step.
        rot_actions[:, 2] = torch.where(
            self.tp.unidirectional_rot, -(rot_actions[:, 2] + 1.0) * 0.5, rot_actions[:, 2]  # [-1, 0]
        )
        rot_actions = rot_actions * self.rot_threshold

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_curr_successes(self, height_threshold, check_rot):
        """Get success mask at current timestep.

        HETERO: ``height_threshold`` is per env in metres and ``check_rot`` a per-env mask (official: one task's
        threshold fraction and one flag; the conversion to metres is in task_table.py).
        """
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = self._held_base_pose()
        target_held_base_pos, target_held_base_quat = self._target_held_base_pose()

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        # HETERO: computed for every env and applied where check_rot is set (no CPU-GPU sync per step)
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        curr_yaw = factory_utils.wrap_yaw(curr_yaw)
        is_rotated = curr_yaw < self.tp.ee_success_yaw
        curr_successes = torch.logical_and(curr_successes, torch.logical_or(is_rotated, ~check_rot))

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        super()._log_factory_metrics(rew_dict, curr_successes)
        # HETERO: the same statistics per task (official: one task per env instance). Episodes cut short by a
        # homo switch are logged under "<key>_truncated" so they do not mix with full-length episodes.
        if torch.any(self.reset_buf):
            suffix = "_truncated" if self._forced_reset else ""
            self._last_success_by_task = {}
            for t, name in enumerate(self.task_names):
                ids = self.env_ids_by_task[t]
                if ids.numel():
                    rate = torch.count_nonzero(curr_successes[ids]) / ids.numel()
                    self.extras[f"successes_{name}{suffix}"] = rate
                    self._last_success_by_task[name] = rate
        for t, name in enumerate(self.task_names):
            ids = self.env_ids_by_task[t]
            if ids.numel():
                self.extras[f"rew_{name}"] = self._last_rew_buf[ids].mean()

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        # HETERO: per-env threshold and rotation check (official: cfg_task.success_threshold, nut_thread only)
        curr_successes = self._get_curr_successes(
            height_threshold=self.tp.success_height_threshold, check_rot=self.tp.success_check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._last_rew_buf = rew_buf  # HETERO: for the per-task reward log
        self._ep_return += rew_buf  # HETERO: per-task episode return, logged at the reset
        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        held_base_pos, held_base_quat = self._held_base_pose()
        target_held_base_pos, target_held_base_quat = self._target_held_base_pose()

        num_keypoints = self.task_table.num_keypoints
        keypoints_held = torch.zeros((self.num_envs, num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, num_keypoints, 3), device=self.device)
        offsets = factory_utils.get_keypoint_offsets(num_keypoints, self.device)
        # HETERO: keypoint scale per env, [num_envs, K, 3] (official: one scale, [K, 3])
        keypoint_offsets = offsets.unsqueeze(0) * self.tp.keypoint_scale.view(-1, 1, 1)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        for idx in range(num_keypoints):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat, held_base_pos, identity, keypoint_offsets[:, idx]
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat, target_held_base_pos, identity, keypoint_offsets[:, idx]
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        # HETERO: coefficients per env
        a0, b0 = self.tp.keypoint_coef_baseline[:, 0], self.tp.keypoint_coef_baseline[:, 1]
        a1, b1 = self.tp.keypoint_coef_coarse[:, 0], self.tp.keypoint_coef_coarse[:, 1]
        a2, b2 = self.tp.keypoint_coef_fine[:, 0], self.tp.keypoint_coef_fine[:, 1]
        # Action penalties.
        # (Official: one norm over all envs, not per env. Its scale is 0 for all three tasks.)
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(
            height_threshold=self.tp.engage_height_threshold,
            check_rot=torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device),
        )

        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.tp.action_penalty_ee_scale,
            "action_grad_penalty": -self.tp.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    # ------------------------------------------------------------------ reset

    def _base_reset_idx(self, env_ids):
        """DirectRLEnv's part of the reset (scene buffers, episode counters)."""
        DirectRLEnv._reset_idx(self, env_ids)

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        # HETERO: required here too, because the task assignment is redrawn for the whole batch
        if len(env_ids) != self.num_envs:
            raise RuntimeError(f"HeteroFactoryEnv resets all envs at once, got {len(env_ids)} of {self.num_envs}")
        steps_done = int(self.episode_length_buf[0])  # HETERO: length of the episode that just ended (all equal)
        self._base_reset_idx(env_ids)

        # HETERO: per task, the episode that just ended (before the tasks change): mean return in extras, and one
        # record per task in episode_log (success at the last step, return, steps, whether a homo switch cut it)
        if self.episode_count > 0:
            suffix = "_truncated" if self._forced_reset else ""
            steps = steps_done
            for t, name in enumerate(self.task_names):
                ids = self.env_ids_by_task[t]
                if ids.numel():
                    ret = self._ep_return[ids].mean()
                    self.extras[f"return_{name}{suffix}"] = ret
                    success = self._last_success_by_task.get(name)
                    self.episode_log.append(dict(
                        episode=self.episode_count, task=name, n_envs=int(ids.numel()),
                        success=float(success) if success is not None else None, ret=float(ret),
                        steps=steps, truncated=bool(self._forced_reset),
                    ))
        self._ep_return[:] = 0.0

        # HETERO: homo "episode" switching: next task after every K complete episodes
        a = self.cfg.assignment
        k = int(getattr(a, "homo_switch_every_episodes", 0) or 0)
        if a.mode == "homo" and k > 0 and self.episode_count > 0 and self.episode_count % k == 0:
            self.homo_task = self.next_homo_task()

        # HETERO: this episode's task per env
        self._set_task_ids(self._next_task_ids())
        self.episode_count += 1
        if a.mode == "homo":
            self.extras["homo_task"] = self.homo_task

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization.

        HETERO: each env's active held and fixed parts go to their official default pose, every other part to
        its parking spot. The small and large gears of gear_mesh envs are left where they are, as in the
        official env; randomize_initial_state seats them on the gear base.
        """
        for slot, (task, role, _attr, _prim) in PART_SLOTS.items():
            active = self.task_ids[env_ids] == self.task_names.index(task)
            if role in ("held", "fixed"):
                act_ids = env_ids[active]
                if len(act_ids):
                    state = self._slot_default_state(slot, act_ids)
                    state[:, 0:3] += self.scene.env_origins[act_ids]
                    state[:, 7:] = 0.0
                    self._slot_write(slot, state, act_ids)
            self._park(slot, env_ids[~active])

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        # HETERO: from the task table, where the official method was evaluated once per task
        return self.tp.held_in_hand_pos.clone(), self.tp.held_in_hand_quat.clone()

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        # HETERO: per-env gripper width (official: cfg_task.held_asset_cfg.diameter / 2 * 1.25)
        gripper_width = self.tp.gripper_width[env_ids]
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width.unsqueeze(-1)  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        tp = self.tp
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # (1.) Randomize fixed asset pose.
        # HETERO: each env's active fixed part (official: self._fixed_asset)
        fixed_state = self._active_default_root_state("fixed", env_ids)
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        # HETERO: per-env noise (official: @ torch.diag(cfg_task.fixed_asset_init_pos_noise))
        fixed_pos_init_rand = fixed_pos_init_rand * tp.fixed_asset_init_pos_noise[env_ids]
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        # HETERO: per-env yaw and range (official: one task's, in degrees, converted with np.deg2rad)
        fixed_orn_init_yaw = torch.deg2rad(tp.fixed_asset_init_orn_deg[env_ids]).unsqueeze(-1)
        fixed_orn_yaw_range = torch.deg2rad(tp.fixed_asset_init_orn_range_deg[env_ids]).unsqueeze(-1)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._write_active_root_state("fixed", fixed_state, env_ids)

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        # HETERO: per-env tip offset from the task table (official: computed here from cfg_task)
        fixed_tip_pos_local = tp.fixed_tip_pos_local.clone()

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            # HETERO: per env (official: += cfg_task.hand_init_pos[2])
            above_fixed_pos[:, 2] += tp.hand_init_pos[:, 2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            # HETERO: per-env noise (official: @ torch.diag(cfg_task.hand_init_pos_noise))
            above_fixed_pos_rand = above_fixed_pos_rand * tp.hand_init_pos_noise[bad_envs]
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # (b) get random orientation facing down
            # HETERO: per env (official: cfg_task.hand_init_orn repeated)
            hand_down_euler = tp.hand_init_orn[bad_envs].clone()

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            # HETERO: per-env noise (official: @ torch.diag(cfg_task.hand_init_orn_noise))
            above_fixed_orn_noise = above_fixed_orn_noise * tp.hand_init_orn_noise[bad_envs]
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
            )

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

            ik_attempt += 1
        # HETERO: every retry steps the whole scene, so the slowest env sets the reset time
        self.last_reset_ik_attempts = ik_attempt
        self.extras["reset_ik_attempts"] = ik_attempt

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        # HETERO: only for envs whose task has them (official: if cfg_task is gear_mesh and add_flanking_gears)
        gear_mask = tp.has_flanking_gears[env_ids]
        if bool(gear_mask.any()):
            gear_ids = env_ids[gear_mask]
            for slot in ("gear_small", "gear_large"):
                gear_state = self._slot_default_state(slot, gear_ids)
                gear_state[:, 0:7] = fixed_state[gear_mask, 0:7]
                gear_state[:, 7:] = 0.0  # vel
                self._slot_write(slot, gear_state, gear_ids)

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        # HETERO: per env (official: if cfg_task is gear_mesh)
        held_asset_pos_noise[:, 2] = torch.where(
            tp.held_noise_z_one_sided, -rand_sample[:, 2], held_asset_pos_noise[:, 2]
        )  # [-1, 0] for gear_mesh

        # HETERO: per-env noise level (official: @ torch.diag(cfg_task.held_asset_pos_noise))
        held_asset_pos_noise = held_asset_pos_noise * tp.held_asset_pos_noise
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        # HETERO: each env's active held part (official: self._held_asset, all envs)
        held_state = self._active_default_root_state("held", torch.arange(self.num_envs, device=self.device))
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._write_active_root_state("held", held_state, torch.arange(self.num_envs, device=self.device))

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
