"""HeteroP2PEnv: the official Play2Perfect ``PreciseAssemblyEnv`` with the four assembly problems in one batch.

Subclass of the official env (``isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py``, commit ff2fc62).
Everything not overridden is inherited unchanged: the action pipeline, wrench randomisation, observations,
rewards, terminations, the retract phase, the tolerance curriculum. ``env.object`` / ``env.hole`` /
``env.goal_viz`` are ``ActiveRigidObject``s (``active_views.py``), so that inherited code reads and writes each
env's active part. Overrides are copies of the official methods in which the single problem's values
(``self._pih_hole_z_offset``, ``self._insert_pos_rel``, ``self._num_total_insertion_goals``) become per-env
values gathered by each env's task; changes are marked ``HETERO``.

Scene: union scene (``p2p_env_cfg.py``): every env holds every task's object, hole and goal marker; the parts of
the tasks an env is not running are parked. Envs reset one by one, as in the official env.

Tasks per env:
* hetero: env i runs task i % num_tasks for the whole run. With SAPG's blocks of 4096 consecutive envs, every
  block holds every task equally.
* homo: every env runs ``homo_task``. ``schedule_homo_switch`` makes every env time out on a chosen step (a
  truncation, so PPO bootstraps the value), and the resets that follow start the next task.

Observations: the official policy and critic inputs, then the task one-hot (RULES.md: shared policy with a task
id). rl_games' SAPG appends its block column after that.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from isaacsimenvs.tasks.play.utils.scene_utils import (
    _bake_usd,
    _convert_urdf_to_usd,
    _log_scene_step,
    _materialize_env_prims,
    _robot_joint_drive_cfg,
    build_robot_articulation_usd_cfg,
)
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import TABLE_HALF_HEIGHT, PreciseAssemblyEnv
from isaacsimenvs.tasks.precise_assembly.scene_utils import REPO_ROOT, _asset_path

from .active_views import ActiveRigidObject
from .p2p_env_cfg import HeteroP2PEnvCfg
from .problem_table import ProblemTable, load_problem_specs


class HeteroP2PEnv(PreciseAssemblyEnv):
    cfg: HeteroP2PEnvCfg

    def __init__(self, cfg: HeteroP2PEnvCfg, render_mode: str | None = None, **kwargs):
        self.task_names = tuple(cfg.problems)
        self.num_tasks = len(self.task_names)
        self.specs = load_problem_specs(self.task_names, cfg.precise_assembly.goal_mode, REPO_ROOT)
        if float(cfg.precise_assembly.random_goal_fraction) != 0.0:
            raise NotImplementedError("random-goal envs are not part of the assembly tasks")
        super().__init__(cfg, render_mode, **kwargs)
        self._init_hetero()

    # ------------------------------------------------------------------ problem configuration

    def _configure_problem(self, cfg: HeteroP2PEnvCfg) -> None:
        """Official ``_configure_problem`` for the first task (it sets the attributes the inherited code expects);
        the per-task values live in ``self.table_problem`` (built in ``_setup_scene``, once the device is known)."""
        cfg.precise_assembly.problem = self.task_names[0]
        super()._configure_problem(cfg)
        if self._num_prelude_goals:
            raise NotImplementedError("goal modes with a prelude are not supported")
        # HETERO: the longest sequence. Only read by log_step_metrics' "all_goals_hit", which _log_peg_metrics
        # (official) then replaces with the per-env count.
        cfg.termination.max_consecutive_successes = max(s.num_goals for s in self.specs)

    # ------------------------------------------------------------------ tasks per env

    def _initial_task_ids(self) -> torch.Tensor:
        a = self.cfg.assignment
        if a.mode == "hetero":
            return torch.arange(self.num_envs, device=self.device) % self.num_tasks
        if a.mode == "homo":
            return torch.full((self.num_envs,), int(a.homo_task), dtype=torch.long, device=self.device)
        raise ValueError(f"unknown assignment mode {a.mode!r}")

    def _set_task_rows(self, env_ids: torch.Tensor, task_ids: torch.Tensor):
        """Set the task of env_ids and every per-env value that depends on it."""
        self.task_ids[env_ids] = task_ids
        self._object_scale_per_env[env_ids] = self.table_problem.object_scale[task_ids]
        self._object_asset_index_per_env[env_ids] = task_ids
        if hasattr(self, "_keypoint_offsets"):  # allocated by the official allocate_state_buffers
            from isaacsimenvs.tasks.play.utils.obs_utils import KEYPOINT_CORNERS

            rew = self.cfg.reward
            corners = torch.tensor(KEYPOINT_CORNERS, device=self.device, dtype=torch.float32)
            # official allocate_state_buffers, for these envs
            self._keypoint_offsets[env_ids] = corners.unsqueeze(0) * (
                self._object_scale_per_env[env_ids] * rew.object_base_size * rew.keypoint_scale * 0.5
            ).unsqueeze(1)
        if hasattr(self, "_object_mass_table"):  # built in _init_hetero
            self._object_mass[env_ids] = self._object_mass_table[task_ids, env_ids]
        self._parked_ok[env_ids] = False  # the kinematic parts must be re-parked at the next reset
        self.task_onehot[env_ids] = torch.nn.functional.one_hot(task_ids, self.num_tasks).float()

    def set_homo_task(self, task: int | str):
        self.homo_task = self.task_names.index(task) if isinstance(task, str) else int(task)

    def next_homo_task(self) -> int:
        """The cycle tight_insertion -> beam step 1 -> beam step 2 -> screwing -> ..."""
        return (self.homo_task + 1) % self.num_tasks

    def schedule_homo_switch(self, after_steps: int, task: int | None = None):
        """Homo, switching at a rollout boundary: after ``after_steps`` more env steps every env times out, and the
        resets that follow start ``task`` (default: the next one)."""
        self._switch_countdown = int(after_steps)
        self._switch_task = self.next_homo_task() if task is None else int(task)

    def pop_samples_per_task(self) -> list[int]:
        out = self._samples_per_task.tolist()
        self._samples_per_task.zero_()
        return out

    # ------------------------------------------------------------------ scene

    def _setup_scene(self) -> None:
        """Official ``scene_utils.setup_scene`` with one object, hole and goal marker per task (HETERO)."""
        assets_cfg = self.cfg.assets
        t0 = time.perf_counter()
        _log_scene_step(t0, f"union scene setup start num_envs={self.num_envs} tasks={self.task_names}")
        self._tmp_asset_dir = tempfile.mkdtemp(prefix="hetero_p2p_assets_")
        usd_work_dir = Path(self._tmp_asset_dir) / "usd"
        bake_root = Path(self._tmp_asset_dir) / "baked_usd"
        usd_work_dir.mkdir(parents=True, exist_ok=True)
        self._object_urdf_paths = [s.object_urdf for s in self.specs]
        self._hole_urdf_paths = [s.receptive_urdf for s in self.specs]
        self._table_urdf_paths = [_asset_path(assets_cfg.table_urdf)]

        object_usd, goal_usd, hole_usd = [], [], []
        for t, s in enumerate(self.specs):
            work = usd_work_dir / f"task{t}"
            work.mkdir(parents=True, exist_ok=True)
            raw = _convert_urdf_to_usd(s.object_urdf, work, fix_base=False)
            # props: official setup_scene
            object_usd.append(_bake_usd(raw, bake_root, f"object{t}", props=dict(
                kinematic_enabled=False, disable_gravity=False, max_depenetration_velocity=1000.0,
                rb_solver_position_iterations=4, rb_solver_velocity_iterations=0, articulation_enabled=False)))
            goal_usd.append(_bake_usd(raw, bake_root, f"goalviz{t}", props=dict(
                kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
                rb_solver_position_iterations=4, rb_solver_velocity_iterations=0), collision_enabled=False))
            hole_usd.append(_bake_usd(_convert_urdf_to_usd(s.receptive_urdf, work / "hole", fix_base=False),
                                      bake_root, f"hole{t}", props=dict(
                kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
                rb_solver_position_iterations=4, rb_solver_velocity_iterations=0)))
        robot_usd = _bake_usd(
            _convert_urdf_to_usd(_asset_path(assets_cfg.robot_urdf), usd_work_dir, fix_base=True,
                                 self_collision=False, joint_drive=_robot_joint_drive_cfg()),
            bake_root, "robot", props=dict(disable_gravity=True, max_depenetration_velocity=1000.0,
                                           enabled_self_collisions=False, solver_position_iterations=8,
                                           solver_velocity_iterations=0), apply_physx_articulation=True)
        table_usd = _bake_usd(
            _convert_urdf_to_usd(_asset_path(assets_cfg.table_urdf), usd_work_dir, fix_base=False),
            bake_root, "table", props=dict(kinematic_enabled=True, disable_gravity=True, articulation_enabled=False))
        _log_scene_step(t0, "converted object/hole/table URDFs")

        _materialize_env_prims(self)

        def rigid(prim: str, usd: str, pos, contact_sensors: bool = False) -> RigidObject:
            # official _pih_rigid_object_cfg, plus a start pose at the part's parking spot so that no two parts
            # overlap before the first reset (HETERO)
            spawn = UsdFileCfg(usd_path=usd, activate_contact_sensors=contact_sensors)
            return RigidObject(RigidObjectCfg(prim_path=f"/World/envs/env_.*/{prim}", spawn=spawn,
                                              init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pos))))

        self.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd))
        self.table = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=UsdFileCfg(usd_path=table_usd)))
        self._objects, self._holes, self._goals = [], [], []
        for t in range(self.num_tasks):
            self._objects.append(rigid(f"Object_{t}", object_usd[t], self._park_local("object", t),
                                       contact_sensors=bool(getattr(self.cfg, "object_contact_sensors", False))))
            self._holes.append(rigid(f"Hole_{t}", hole_usd[t], self._park_local("hole", t)))
            self._goals.append(rigid(f"GoalViz_{t}", goal_usd[t], self._park_local("goal", t)))
        _log_scene_step(t0, "spawned robot/table and every task's object/hole/goalviz")

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        for t in range(self.num_tasks):
            self.scene.rigid_objects[f"object_{t}"] = self._objects[t]
            self.scene.rigid_objects[f"hole_{t}"] = self._holes[t]
            self.scene.rigid_objects[f"goal_viz_{t}"] = self._goals[t]

        # HETERO: the official env.object / env.hole / env.goal_viz
        self.object = ActiveRigidObject(self, self._objects, dynamic=True)
        self.hole = ActiveRigidObject(self, self._holes, dynamic=False)
        self.goal_viz = ActiveRigidObject(self, self._goals, dynamic=False)

        # HETERO: per-task table and the initial task of every env
        self.table_problem = ProblemTable(self.specs, self.device)
        self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_onehot = torch.zeros((self.num_envs, self.num_tasks), device=self.device)
        self._object_scale_per_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._object_asset_index_per_env = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._parked_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._set_task_rows(all_ids, self._initial_task_ids())

        if self.scene._default_env_origins is None:  # official
            self.scene.clone_environments(copy_from_source=False)
            _log_scene_step(t0, "cloned environments (replicate_physics=True path)")

    def _park_local(self, kind: str, t: int):
        c = self.cfg
        x, y = c.object_park_xy[t]
        if kind == "object":
            return (x, y, c.object_park_z + 0.5 * t)
        if kind == "hole":
            return (x, y, c.hole_park_z)
        return (x, y, c.goal_park_z)

    def _park_poses(self, kind: str) -> torch.Tensor:
        """[tasks, envs, 7] world parking poses."""
        out = torch.zeros((self.num_tasks, self.num_envs, 7), device=self.device)
        for t in range(self.num_tasks):
            out[t, :, 0:3] = torch.tensor(self._park_local(kind, t), device=self.device) + self.scene.env_origins
            out[t, :, 3] = 1.0
        return out

    # ------------------------------------------------------------------ after the official __init__

    def _init_hetero(self):
        # the official __init__ took the first task's values; replace them with per-env ones
        tp = self.table_problem
        self._insert_pos_rel_t = tp.insert_pos_rel
        self._insert_quat_wxyz_t = tp.insert_quat_wxyz
        self.env_max_goals[:] = tp.num_goals[self.task_ids]
        self.prev_episode_env_max_goals[:] = self.env_max_goals
        # object masses of every task's object in every env (for the official wrench randomisation)
        self._object_mass_table = torch.stack(
            [v.data.default_mass[:, 0:1].to(self.device) for v in self._objects], dim=0)  # [tasks, envs, 1]
        self._object_mass = self._object_mass_table[self.task_ids, torch.arange(self.num_envs, device=self.device)]
        self._park = {k: self._park_poses(k) for k in ("object", "hole", "goal")}

        # homo switching and per-task bookkeeping
        self.homo_task = int(self.cfg.assignment.homo_task)
        self._switch_countdown = None
        self._switch_task = None
        self._forced_reset = False
        self._samples_per_task = torch.zeros(self.num_tasks, dtype=torch.long, device=self.device)
        self._ep_return = torch.zeros(self.num_envs, device=self.device)
        self._ep_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.set_block_size(self.num_envs)  # the training script sets SAPG's block size

        # HETERO: policy and critic inputs end with the task one-hot. The official buffers (observation delay
        # queue) keep the official widths: the one-hot is appended after them.
        self.cfg.observation_space = int(self.cfg.observation_space) + self.num_tasks
        self.cfg.state_space = int(self.cfg.state_space) + self.num_tasks
        self._configure_gym_env_spaces()
        print(f"[HeteroP2PEnv] tasks={self.task_names} mode={self.cfg.assignment.mode} "
              f"goals={tp.num_goals.tolist()} obs={self.cfg.observation_space} state={self.cfg.state_space}",
              flush=True)

    # ------------------------------------------------------------------ step hooks

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = super()._get_observations()
        obs["policy"] = torch.cat([obs["policy"], self.task_onehot], dim=-1)  # HETERO
        obs["critic"] = torch.cat([obs["critic"], self.task_onehot], dim=-1)  # HETERO
        return obs

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        # HETERO: samples collected per task (this step's tasks, before any reset)
        self._samples_per_task += torch.bincount(self.task_ids, minlength=self.num_tasks)
        self._forced_reset = False
        if self._switch_countdown is not None:
            self._switch_countdown -= 1
            if self._switch_countdown <= 0:
                truncated = torch.ones_like(truncated)
                self._forced_reset = True
                self.homo_task = self._switch_task
                self._switch_countdown = None
        self._last_terminated = terminated
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        self._ep_return += reward
        self._ep_steps += 1
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "_ep_steps"):
            self._log_episodes(env_ids)
            # HETERO: homo switch takes effect at the reset that follows it
            if self.cfg.assignment.mode == "homo":
                new = torch.full_like(env_ids, self.homo_task)
                if bool((self.task_ids[env_ids] != new).any()):
                    self._set_task_rows(env_ids, new)  # env_max_goals follows in _reset_peg_episode
        super()._reset_idx(env_ids)
        # HETERO: the parts of the other tasks wait at their parking spots
        if hasattr(self, "_park"):
            # dynamic objects at every reset (their anti-gravity force leaves a tiny drift); kinematic holes and
            # goal markers only after a task change
            self.object.park(env_ids, self._park["object"])
            stale = env_ids[~self._parked_ok[env_ids]]
            if stale.numel():
                self.hole.park(stale, self._park["hole"])
                self.goal_viz.park(stale, self._park["goal"])
                self._parked_ok[stale] = True

    EP_FIELDS = ("episodes", "success", "goal_ratio", "retract_success", "return", "steps", "forced")

    def _log_episodes(self, env_ids: torch.Tensor):
        """Add the episodes ending now to per-(task, block) sums (``pop_episode_stats``). An episode cut by a homo
        switch (``forced``) is counted apart and left out of the success sums."""
        ran = env_ids[self._ep_steps[env_ids] > 0]
        if ran.numel():
            tids = self.task_ids[ran]
            goals = self.env_max_goals[ran]
            succ = self._successes[ran]
            natural = self._last_terminated[ran] | (self.episode_length_buf[ran] >= self.max_episode_length)
            forced = (~natural) & bool(self._forced_reset)
            keep = (~forced).float()
            vals = torch.stack([
                keep, (succ >= goals).float() * keep, (succ.float() / goals.clamp_min(1).float()) * keep,
                self.retract_succeeded[ran].float() * keep, self._ep_return[ran] * keep,
                self._ep_steps[ran].float() * keep, forced.float()], dim=-1)
            row = tids * self._num_blocks + torch.div(ran, self.block_size, rounding_mode="floor")
            self._ep_agg.index_add_(0, row, vals)
        self._ep_return[env_ids] = 0.0
        self._ep_steps[env_ids] = 0

    def set_block_size(self, block_size: int):
        """Group episode statistics by blocks of ``block_size`` consecutive envs (SAPG's exploration blocks)."""
        if self.num_envs % int(block_size):
            raise ValueError(f"num_envs {self.num_envs} is not a multiple of block size {block_size}")
        self.block_size = int(block_size)
        self._num_blocks = self.num_envs // self.block_size
        self._ep_agg = torch.zeros((self.num_tasks * self._num_blocks, len(self.EP_FIELDS)), device=self.device)

    def pop_episode_stats(self) -> dict:
        """Sums since the last call: {task: {block: {field: value}}}, plus each task over all blocks ("all")."""
        agg = self._ep_agg.view(self.num_tasks, self._num_blocks, -1).cpu().clone()  # .cpu() is no copy on CPU
        self._ep_agg.zero_()
        out = {}
        for t, name in enumerate(self.task_names):
            d = {str(b): dict(zip(self.EP_FIELDS, agg[t, b].tolist())) for b in range(self._num_blocks)}
            d["all"] = dict(zip(self.EP_FIELDS, agg[t].sum(0).tolist()))
            out[name] = d
        return out

    # ------------------------------------------------------------------ official overrides

    def _reset_peg_episode(self, env_ids: torch.Tensor) -> None:
        """Official ``_reset_peg_episode`` (no prelude, no random-goal envs) with each env's own hole height and
        sub-goal count (HETERO)."""
        n = env_ids.numel()
        pih_cfg = self.cfg.precise_assembly
        env_origins = self.scene.env_origins[env_ids]
        tids = self.task_ids[env_ids]

        self.env_max_goals[env_ids] = self.table_problem.num_goals[tids]  # HETERO

        table_top_z = self._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT

        hole_x_min, hole_x_max = (float(v) for v in pih_cfg.hole_x_range)
        hole_y_min, hole_y_max = (float(v) for v in pih_cfg.hole_y_range)
        self.hole_pos[env_ids, 0] = torch.empty(n, device=self.device).uniform_(hole_x_min, hole_x_max)
        self.hole_pos[env_ids, 1] = torch.empty(n, device=self.device).uniform_(hole_y_min, hole_y_max)
        self.hole_pos[env_ids, 2] = table_top_z + self.table_problem.hole_z_offset[tids]  # HETERO

        yaw_range_deg = float(pih_cfg.hole_yaw_range_deg)
        if yaw_range_deg > 0.0:
            import math

            yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * yaw_range_deg * (math.pi / 180.0)
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).expand(n, -1)
            hole_quat = quat_from_angle_axis(yaw, z_axis)
        else:
            hole_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device,
                                     dtype=torch.float32).unsqueeze(0).expand(n, -1).contiguous()
        self.hole_quat_wxyz[env_ids] = hole_quat
        hole_pose = torch.cat([self.hole_pos[env_ids] + env_origins, hole_quat], dim=-1)
        self.hole.write_root_pose_to_sim(hole_pose, env_ids=env_ids)
        self.hole.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        self.goal_pos_obs_noise[env_ids] = 0.0
        self.goal_yaw_obs_noise[env_ids] = 0.0
        noise = float(pih_cfg.goal_xy_obs_noise)
        if noise > 0.0:
            self.goal_pos_obs_noise[env_ids, 0:2] = torch.empty(n, 2, device=self.device).uniform_(-noise, noise)
        yaw_noise_deg = float(pih_cfg.goal_yaw_obs_noise_deg)
        if yaw_noise_deg > 0.0:
            import math

            yaw_max = yaw_noise_deg * (math.pi / 180.0)
            self.goal_yaw_obs_noise[env_ids] = torch.empty(n, device=self.device).uniform_(-yaw_max, yaw_max)

        self.retract_phase[env_ids] = False
        self.retract_succeeded[env_ids] = False
        self._just_entered_retract[env_ids] = False
        self._just_retracted[env_ids] = False

        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids, is_first_goal=True)

    def _write_goal_pose(self, env_ids: torch.Tensor, is_first_goal: bool = False) -> None:
        """Official ``_write_goal_pose`` for insertion envs without a prelude, with each env's task's sub-goal
        poses (HETERO)."""
        n = env_ids.numel()
        env_origins = self.scene.env_origins[env_ids]
        subgoal_idx = (self._successes[env_ids] % self.env_max_goals[env_ids]).long()
        tids = self.task_ids[env_ids]
        hole_q = self.hole_quat_wxyz[env_ids]
        insert_pos = self._insert_pos_rel_t[tids, subgoal_idx]  # HETERO
        insert_q = self._insert_quat_wxyz_t[tids, subgoal_idx]  # HETERO
        pos_local = self.hole_pos[env_ids] + quat_apply(hole_q, insert_pos)
        quat = quat_mul(hole_q, insert_q)
        pose = torch.cat([pos_local + env_origins, quat], dim=-1)
        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
