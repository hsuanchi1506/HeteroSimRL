"""Pose recording of the union-scene Play2Perfect env for ``scripts/render_offline.py`` (same file format as
``hetero_factory/pose_record.py``, whose mesh export is reused).

Recorded assets: the robot, the table, and every task's object, receptive part (hole) and goal marker. A part
that is parked in an env (another task's) is written far below the scene, so the renderer never shows it. The
camera target is each env's hole; the success flag is "every sub-goal reached" (the official offline metric).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hetero_factory.pose_record import ROBOT_RGBA, TABLE_RGBA, export_meshes

OBJECT_RGBA = ((0.31, 0.63, 1.0, 1.0), (1.0, 0.67, 0.24, 1.0), (0.45, 0.80, 0.45, 1.0), (0.75, 0.43, 1.0, 1.0))
HOLE_RGBA = (0.45, 0.47, 0.52, 1.0)
GOAL_ALPHA = 0.25
HIDDEN = (0.0, 0.0, -50.0)


class P2PPoseRecorder:
    def __init__(self, env):
        import omni.usd

        import isaaclab.sim as sim_utils

        self.env = env
        stage = omni.usd.get_context().get_stage()
        self.assets = {"robot": env.robot, "table": env.table}
        self.task_of = {"robot": None, "table": None}
        for t in range(env.num_tasks):
            for kind, views in (("object", env._objects), ("hole", env._holes), ("goal", env._goals)):
                name = f"{kind}_{t}"
                self.assets[name] = views[t]
                self.task_of[name] = t
        self.origins = env.scene.env_origins
        self.meta = dict(num_envs=env.num_envs, task_names=list(env.task_names), assets={}, meshes=[],
                         dt=float(env.step_dt), scene="p2p_union")
        self._mesh_arrays = []
        for name, asset in self.assets.items():
            root = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)[0]
            t = self.task_of[name]
            if name == "robot":
                default, force = ROBOT_RGBA, None
            elif name == "table":
                default, force = TABLE_RGBA, TABLE_RGBA
            elif name.startswith("object"):
                default = force = OBJECT_RGBA[t % len(OBJECT_RGBA)]
            elif name.startswith("hole"):
                default = force = HOLE_RGBA
            else:  # goal marker: the object's mesh, faint
                c = OBJECT_RGBA[t % len(OBJECT_RGBA)]
                default = force = (c[0], c[1], c[2], GOAL_ALPHA)
            meshes = export_meshes(stage, root, None, set(asset.body_names), default, force)
            self.meta["assets"][name] = dict(task=None if t is None else env.task_names[t], role=name.split("_")[0],
                                             body_names=list(asset.body_names),
                                             rows_env=list(range(env.num_envs)), source_prim=root)
            for body, v, f, rgba in meshes:
                self.meta["meshes"].append(dict(asset=name, body=body, rgba=rgba, nv=len(v), nf=len(f)))
                self._mesh_arrays.append((v, f))
        self.pos = {n: [] for n in self.assets}
        self.quat = {n: [] for n in self.assets}
        self.cam_target, self.success, self.time = [], [], []

    @torch.inference_mode()
    def record(self, time_s: float):
        env = self.env
        hidden = torch.tensor(HIDDEN, device=env.device)
        for name, asset in self.assets.items():
            pos = asset.data.body_pos_w - self.origins.unsqueeze(1)  # [envs, bodies, 3]
            t = self.task_of[name]
            if t is not None:
                parked = env.task_ids != t
                pos = torch.where(parked.view(-1, 1, 1), hidden, pos)
            self.pos[name].append(pos.cpu().numpy().astype(np.float32))
            self.quat[name].append(asset.data.body_quat_w.cpu().numpy().astype(np.float32))  # w, x, y, z
        self.cam_target.append(env.hole_pos.cpu().numpy().astype(np.float32))  # env frame
        self.success.append((env._successes >= env.env_max_goals).cpu().numpy())
        self.time.append(float(time_s))

    def save(self, path, extra: dict | None = None) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            mesh_verts=np.concatenate([v for v, _ in self._mesh_arrays]),
            mesh_faces=np.concatenate([f for _, f in self._mesh_arrays]),
            cam_target=np.stack(self.cam_target), success=np.stack(self.success), time=np.asarray(self.time),
        )
        for name in self.assets:
            arrays[f"pos__{name}"] = np.stack(self.pos[name])
            arrays[f"quat__{name}"] = np.stack(self.quat[name])
        meta = dict(self.meta, **(extra or {}))
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
        return path
