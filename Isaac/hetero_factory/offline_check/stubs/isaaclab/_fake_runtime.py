"""A fake Isaac Lab runtime for offline logic checks: no physics, bodies move only when written.

Records every root-pose write per asset so tests can check which parts were moved for which envs.
The robot fingertip stays where the test puts it (tests patch the reset IK)."""
import math
import re

import torch

STATE = {"sim": None, "scene": None, "assets": [], "spawned": []}
ROBOT_BODIES = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
                "panda_link6", "panda_link7", "force_sensor", "panda_hand", "panda_leftfinger",
                "panda_rightfinger", "panda_fingertip_centered"]


class FakeView:
    def __init__(self, art):
        self.art = art

    def get_jacobians(self):
        g = torch.Generator().manual_seed(0)
        n, b = self.art.n, len(self.art.body_names)
        return torch.randn((n, b - 1, 6, 9), generator=g) * 0.3 + torch.eye(6, 9).expand(n, b - 1, 6, 9)

    def get_generalized_mass_matrices(self):
        return torch.eye(9).expand(self.art.n, 9, 9).clone()

    def get_material_properties(self):
        return self.art.materials.clone()

    def set_material_properties(self, m, env_ids):
        self.art.materials[env_ids.long()] = m[env_ids.long()]

    def get_inertias(self):
        return torch.ones((self.art.n, len(self.art.body_names), 9))

    def set_inertias(self, x, env_ids):
        pass


class FakeData:
    pass


class FakeArticulation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_robot = cfg.prim_path.endswith("/Robot")
        self.body_names = ROBOT_BODIES if self.is_robot else ["body"]
        self.num_joints = 9 if self.is_robot else 0
        self.writes = []  # (env id list, written positions)
        STATE["assets"].append(self)

    def _envs(self, n):
        """Envs holding this prim: all, unless it was spawned with an env alternation like env_(0|3|6). Such
        subsets are returned in DEScending env order, so code that assumes rows == ascending env ids fails."""
        name = self.cfg.prim_path.rsplit("/", 1)[1]
        for path in STATE["spawned"]:
            root, leaf = path.rsplit("/", 1)
            token = root.rsplit("/", 1)[1]
            if leaf == name and token != "env_.*":
                return sorted((i for i in range(n) if re.fullmatch(token, f"env_{i}")), reverse=True)
        return list(range(n))

    def _initialize(self, n, origins):
        envs = self._envs(n)
        self.env_ids_of_rows = envs
        origins = origins[envs]
        n = len(envs)
        self.n = n
        d = self._data = self.data = FakeData()
        init = self.cfg.init_state
        pos = torch.tensor(getattr(init, "pos", (0.0, 0.0, 0.0)), dtype=torch.float32)
        rot = torch.tensor(getattr(init, "rot", (1.0, 0.0, 0.0, 0.0)), dtype=torch.float32)
        d.default_root_state = torch.zeros((n, 13))
        d.default_root_state[:, 0:3] = pos
        d.default_root_state[:, 3:7] = rot
        d.root_pos_w = pos + origins.clone()
        d.root_quat_w = rot.expand(n, 4).clone()
        b = len(self.body_names)
        d.body_pos_w = torch.zeros((n, b, 3)) + origins[:, None, :]
        d.body_quat_w = torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(n, b, 4).clone()  # hand pointing down
        d.body_lin_vel_w = torch.zeros((n, b, 3))
        d.body_ang_vel_w = torch.zeros((n, b, 3))
        d.joint_pos = torch.zeros((n, 9))
        d.joint_vel = torch.zeros((n, 9))
        d.default_joint_pos = torch.zeros((n, 9))
        d._sim_timestamp = 0.0
        self.materials = torch.full((n, 1, 3), 0.5)
        self.root_physx_view = FakeView(self)
        name = self.cfg.prim_path.rsplit("/", 1)[1]
        self.root_physx_view.prim_paths = [f"/World/envs/env_{i}/{name}" for i in envs]

    def write_root_pose_to_sim(self, pose, env_ids=None):
        ids = torch.arange(self.n) if env_ids is None else env_ids.long()
        assert pose.shape == (len(ids), 7), (pose.shape, len(ids))
        self.data.root_pos_w[ids] = pose[:, 0:3]
        self.data.root_quat_w[ids] = pose[:, 3:7]
        self.writes.append((ids.tolist(), pose[:, 0:3].clone()))

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        ids = torch.arange(self.n) if env_ids is None else env_ids.long()
        assert vel.shape == (len(ids), 6), (vel.shape, len(ids))

    def reset(self, env_ids=None):
        pass

    def write_joint_state_to_sim(self, pos, vel, env_ids=None):
        ids = torch.arange(self.n) if env_ids is None else env_ids.long()
        if pos.shape[0] == self.n and len(ids) != self.n:
            pos, vel = pos[ids], vel[ids]
        self.data.joint_pos[ids] = pos
        self.data.joint_vel[ids] = vel

    def set_joint_position_target(self, t, env_ids=None):
        pass

    def set_joint_effort_target(self, t, env_ids=None):
        pass


class FakeSim:
    def __init__(self, cfg):
        self.dt = cfg.sim.dt
        self.steps = 0
        self.gravity_calls = []
        self.physics_sim_view = self

    def set_gravity(self, g):
        self.gravity_calls.append(tuple(g))

    def step(self, render=False):
        self.steps += 1
        for a in STATE["assets"]:
            a.data._sim_timestamp += self.dt

    def get_physics_dt(self):
        return self.dt


class FakeScene:
    def __init__(self, n):
        self.num_envs = n
        side = math.ceil(math.sqrt(n))
        self.env_origins = torch.tensor([[2.0 * (i // side), 2.0 * (i % side), 0.0] for i in range(n)])
        self.articulations = {}

    def clone_environments(self, copy_from_source=False):
        self.cloned = True  # assets are initialised after _setup_scene, as Isaac Lab does at sim start

    def filter_collisions(self):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def reset(self, env_ids=None):
        pass


class FakeDirectRLEnv:
    """Mirrors the parts of isaaclab.envs.DirectRLEnv (v2.3.2 == main) that FactoryEnv relies on."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        STATE["assets"].clear()
        STATE["spawned"].clear()
        self.cfg = cfg
        self.device = "cpu"
        self.num_envs = cfg.scene.num_envs
        self.sim = STATE["sim"] = FakeSim(cfg)
        self.scene = STATE["scene"] = FakeScene(self.num_envs)
        self.physics_dt = cfg.sim.dt
        self._setup_scene()
        for a in STATE["assets"]:  # Isaac Lab initialises assets when the simulation starts
            a._initialize(self.num_envs, self.scene.env_origins)
        self.max_episode_length = math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool)
        self.extras = {}
        self.actions = torch.zeros((self.num_envs, cfg.action_space))
        self._sim_step_counter = 0
        self.common_step_counter = 0

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs))
        return self._get_observations(), self.extras

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        self.scene.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
