"""``ActiveRigidObject``: stands in for the official env's single ``env.object`` / ``env.hole`` / ``env.goal_viz``.

In the union scene every env holds one object, one receptive part (hole) and one goal marker per task, each task's
parts being their own Isaac Lab ``RigidObject`` with one instance per env. The official code reads and writes
``env.object.data.root_pos_w``, ``env.object.write_root_pose_to_sim(...)`` and so on (about fifteen places in
``isaacsimenvs/tasks/play/utils`` and ``precise_assembly_env.py``). This class offers exactly those members:
reads return, for every env, the part of that env's task (``env.task_ids``); writes go only to that part. So the
official observation, reward, reset and termination code runs unchanged.

Parked parts (the other tasks' parts of an env) are written by ``park``. Dynamic parked parts (objects) hang in
the air: ``set_external_force_and_torque`` gives them a constant upward force equal to their weight, so they
neither fall nor touch anything. The official code sets the external wrench of the object at every policy step
(``apply_wrench_dr``), which is where the parked force is added.
"""

from __future__ import annotations

import torch


class _ActiveData:
    """The ``.data`` members the official code reads, each env taking its task's row."""

    def __init__(self, owner: "ActiveRigidObject"):
        self._o = owner

    def _gather(self, attr: str) -> torch.Tensor:
        vals = torch.stack([getattr(v.data, attr) for v in self._o.views], dim=0)  # [tasks, envs, ...]
        tids = self._o.env.task_ids.to(vals.device)
        return vals[tids, torch.arange(vals.shape[1], device=vals.device)]

    @property
    def root_pos_w(self):
        return self._gather("root_pos_w")

    @property
    def root_quat_w(self):
        return self._gather("root_quat_w")

    @property
    def root_lin_vel_w(self):
        return self._gather("root_lin_vel_w")

    @property
    def root_ang_vel_w(self):
        return self._gather("root_ang_vel_w")

    @property
    def root_state_w(self):
        return self._gather("root_state_w")

    @property
    def default_mass(self):
        return self._gather("default_mass")


class _MaterialsAdapter:
    """``root_physx_view`` for ``apply_physx_material_properties``: it reads one materials tensor, fills it per env,
    and writes it back. Each task's view has its own shape count, so the per-env values (first shape) are copied
    to every shape of every task's part, which is what the official function does for its single part."""

    def __init__(self, views):
        self._views = views

    def get_material_properties(self):
        return self._views[0].root_physx_view.get_material_properties()[:, :1].clone()

    def set_material_properties(self, materials, env_ids):
        for v in self._views:
            m = v.root_physx_view.get_material_properties()
            m[:] = materials[:, :1]
            v.root_physx_view.set_material_properties(m, env_ids)


class ActiveRigidObject:
    def __init__(self, env, views: list, dynamic: bool):
        self.env = env
        self.views = list(views)
        self.dynamic = dynamic
        self.data = _ActiveData(self)
        self.root_physx_view = _MaterialsAdapter(self.views)
        self._parked_force = None  # [tasks, envs, 1, 3], built on first use (masses are known after sim start)

    @property
    def num_instances(self) -> int:
        return self.views[0].num_instances

    @property
    def device(self):
        return self.views[0].device

    def _env_ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device)
        return torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)

    def _split(self, env_ids):
        """(task, mask over env_ids) for every task that has envs in env_ids."""
        tids = self.env.task_ids[env_ids]
        for t in range(len(self.views)):
            m = tids == t
            if bool(m.any()):
                yield t, m

    # ---- writes the official code makes (active part only)
    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None):
        env_ids = self._env_ids(env_ids)
        for t, m in self._split(env_ids):
            self.views[t].write_root_pose_to_sim(root_pose[m], env_ids=env_ids[m])

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None):
        env_ids = self._env_ids(env_ids)
        for t, m in self._split(env_ids):
            self.views[t].write_root_velocity_to_sim(root_velocity[m], env_ids=env_ids[m])

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids=None):
        self.write_root_pose_to_sim(root_state[:, 0:7], env_ids)
        self.write_root_velocity_to_sim(root_state[:, 7:13], env_ids)

    def set_external_force_and_torque(self, forces, torques, positions=None, body_ids=None, env_ids=None,
                                      is_global=False):
        """The official call (all envs, world frame): active parts get the given wrench, parked dynamic parts
        their anti-gravity force, parked kinematic parts nothing."""
        if env_ids is not None or positions is not None or body_ids is not None:
            raise NotImplementedError("only the official call (all envs, no positions, all bodies) is supported")
        if self.dynamic and not is_global:
            raise NotImplementedError("the parked anti-gravity force is given in the world frame")
        parked = self.parked_force()
        for t, v in enumerate(self.views):
            active = (self.env.task_ids == t).view(-1, 1, 1)
            f = torch.where(active, forces, parked[t]) if self.dynamic else forces * active
            tq = torques * active
            v.set_external_force_and_torque(forces=f, torques=tq, is_global=is_global)

    def parked_force(self) -> torch.Tensor:
        if self._parked_force is None:
            g = torch.tensor(self.env.cfg.sim.gravity, dtype=torch.float32, device=self.env.device)
            out = []
            for v in self.views:
                mass = v.data.default_mass[:, 0:1].to(self.env.device)  # [envs, 1], as the official _object_mass
                out.append((-g).view(1, 1, 3) * mass.view(-1, 1, 1) if self.dynamic else torch.zeros(
                    (self.env.num_envs, 1, 3), device=self.env.device))
            self._parked_force = torch.stack(out, dim=0)
        return self._parked_force

    # ---- parking (not in the official code)
    def park(self, env_ids: torch.Tensor, park_pose: torch.Tensor):
        """Put the parts of the tasks env_ids are NOT running at their parking poses, at rest.
        ``park_pose``: [tasks, envs, 7] world poses (w, x, y, z quaternion)."""
        env_ids = self._env_ids(env_ids)
        tids = self.env.task_ids[env_ids]
        for t, v in enumerate(self.views):
            m = tids != t
            if bool(m.any()):
                ids = env_ids[m]
                v.write_root_pose_to_sim(park_pose[t][ids], env_ids=ids)
                if self.dynamic:  # kinematic parts have no velocity (PhysX's CPU pipeline rejects setting one)
                    v.write_root_velocity_to_sim(torch.zeros((ids.numel(), 6), device=self.env.device), env_ids=ids)
