"""Record what PhysX simulated, to draw it later without Isaac Sim's renderer (scripts/render_offline.py).

Why: on gputw1 (driver 595.84), Isaac Sim 5.1's RTX renderer segfaults in ``librtx.scenedb.plugin.so`` as soon
as a camera-enabled app starts, even with an empty scene. NVIDIA validates Isaac Sim 5.1 only with the 580
production driver; a driver change is up to the machine's admins. Simulation without cameras is unaffected.

What is saved (one ``.npz``): the visible meshes of every asset, read once from the USD stage and expressed in
the frame of the rigid body that carries them (static meshes such as the table: in the env frame), and, per
recorded step, every body's pose in its env's frame, the camera target (the fixed part's tip) and the success
flag per env. Runs inside Isaac Sim (needs ``pxr``); the renderer itself needs only numpy and MuJoCo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

# fallback colours (RGBA) when the USD has no readable material colour
ROBOT_RGBA = (0.82, 0.82, 0.84, 1.0)
TABLE_RGBA = (0.55, 0.45, 0.35, 1.0)
# parts are coloured by role so the viewer sees at once which part is in the hand
HELD_RGBA = {"peg_insert": (0.31, 0.63, 1.0, 1.0), "gear_mesh": (1.0, 0.67, 0.24, 1.0), "nut_thread": (0.75, 0.43, 1.0, 1.0)}
FIXED_RGBA = (0.45, 0.47, 0.52, 1.0)
GEAR_SIDE_RGBA = (0.62, 0.55, 0.42, 1.0)


def _env_of(path: str) -> int:
    return int(re.search(r"/env_(\d+)(/|$)", path).group(1))


def _rigid_part(m):
    """A Gf.Matrix4d without scale and shear (the pose PhysX reports for a body has none)."""
    m = m.RemoveScaleShear()
    return np.array(m, dtype=np.float64)


def _material_rgba(prim):
    """Diffuse colour of the bound material (UsdPreviewSurface or OmniPBR), else displayColor, else None."""
    from pxr import Usd, UsdGeom, UsdShade

    try:
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    except Exception:  # noqa: BLE001
        mat = None
    if mat:
        for p in Usd.PrimRange(mat.GetPrim()):
            shader = UsdShade.Shader(p)
            if not shader:
                continue
            for name in ("diffuseColor", "diffuse_color_constant"):
                inp = shader.GetInput(name)
                if inp and inp.Get() is not None:
                    c = inp.Get()
                    return (float(c[0]), float(c[1]), float(c[2]), 1.0)
    colors = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
    if colors:
        return (float(colors[0][0]), float(colors[0][1]), float(colors[0][2]), 1.0)
    return None


def export_meshes(stage, root_path: str, frame_path: str | None, body_names, default_rgba, force_rgba=None):
    """Visible meshes under ``root_path`` as (body name, vertices [V, 3], triangles [F, 3], rgba).

    Each mesh is expressed in the frame of its nearest ancestor with a rigid body (that body's name is
    returned), or, when ``frame_path`` is given, in that prim's frame (body name None). Faces are fanned into
    triangles; left-handed meshes get their winding reversed so normals point outwards.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"no prim at {root_path}")
    frame_m = _rigid_part(UsdGeom.Xformable(stage.GetPrimAtPath(frame_path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())) if frame_path else None
    out = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        if frame_m is not None:
            body, ref = None, frame_m
        else:
            anc = prim
            while anc.IsValid() and not anc.HasAPI(UsdPhysics.RigidBodyAPI):
                anc = anc.GetParent()
            if not anc.IsValid() or anc.GetName() not in body_names:
                continue  # not carried by one of this asset's bodies
            body = anc.GetName()
            ref = _rigid_part(UsdGeom.Xformable(anc).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        m = np.array(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
        rel = m @ np.linalg.inv(ref)  # USD uses row vectors: p_ref = p_local @ M_mesh @ inv(M_ref)
        p = np.asarray(pts, dtype=np.float64)
        v = (np.c_[p, np.ones(len(p))] @ rel)[:, :3]
        idx = np.asarray(idx, dtype=np.int64)
        tris, start = [], 0
        for c in np.asarray(counts, dtype=np.int64):
            f = idx[start:start + c]
            tris.extend((f[0], f[k], f[k + 1]) for k in range(1, c - 1))
            start += c
        tris = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            tris = tris[:, ::-1]
        rgba = force_rgba or _material_rgba(prim) or default_rgba
        out.append((body, v.astype(np.float32), tris, tuple(float(x) for x in rgba)))
    return out


class PoseRecorder:
    """Meshes once, then body poses per call to ``record``; ``save`` writes ``<path>.npz``.

    ``env`` is the unwrapped HeteroFactoryEnv / HeteroFactoryListEnv (``_robot``, ``_slots``).
    """

    def __init__(self, env, part_slots):
        import omni.usd

        import isaaclab.sim as sim_utils

        self.env = env
        stage = omni.usd.get_context().get_stage()
        self.assets = {"robot": env._robot, **env._slots}
        self.origins = env.scene.env_origins
        self.meta = dict(num_envs=env.num_envs, task_names=list(env.task_names), assets={}, meshes=[],
                         dt=float(env.step_dt))
        self._mesh_arrays = []
        self.rows_env = {}
        for name, asset in self.assets.items():
            paths = list(getattr(asset.root_physx_view, "prim_paths", []) or [])
            rows_env = [_env_of(p) for p in paths] if paths else list(range(env.num_envs))
            self.rows_env[name] = torch.tensor(rows_env, device=env.device)
            root = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)[0]
            if name == "robot":
                default, force = ROBOT_RGBA, None
                info = dict(task=None, role="robot")
            else:
                task, role, _attr, _prim = part_slots[name]
                force = HELD_RGBA[task] if role == "held" else FIXED_RGBA if role == "fixed" else GEAR_SIDE_RGBA
                default = force
                info = dict(task=task, role=role)
            meshes = export_meshes(stage, root, None, set(asset.body_names), default, force)
            self.meta["assets"][name] = dict(info, body_names=list(asset.body_names), rows_env=rows_env,
                                             source_prim=root)
            self._add_meshes(name, meshes)
        # static: the table of env 0, in env 0's frame (every env has the same table at the same place)
        table_root = sim_utils.find_matching_prim_paths("/World/envs/env_0/Table")
        if table_root:
            self._add_meshes("table", export_meshes(stage, table_root[0], "/World/envs/env_0", set(), TABLE_RGBA))
        self.pos = {n: [] for n in self.assets}
        self.quat = {n: [] for n in self.assets}
        self.cam_target, self.success, self.time = [], [], []

    def _add_meshes(self, asset, meshes):
        for body, v, f, rgba in meshes:
            self.meta["meshes"].append(dict(asset=asset, body=body, rgba=rgba, nv=len(v), nf=len(f)))
            self._mesh_arrays.append((v, f))

    @torch.inference_mode()
    def record(self, time_s: float, success):
        for name, asset in self.assets.items():
            origin = self.origins[self.rows_env[name]].unsqueeze(1)  # [rows, 1, 3]
            self.pos[name].append((asset.data.body_pos_w - origin).cpu().numpy().astype(np.float32))
            self.quat[name].append(asset.data.body_quat_w.cpu().numpy().astype(np.float32))  # w, x, y, z
        self.cam_target.append(self.env.fixed_pos_obs_frame.cpu().numpy().astype(np.float32))  # env frame
        self.success.append(np.asarray(success, dtype=bool))
        self.time.append(float(time_s))

    def save(self, path, extra: dict | None = None) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            mesh_verts=np.concatenate([v for v, _ in self._mesh_arrays]) if self._mesh_arrays else np.zeros((0, 3)),
            mesh_faces=np.concatenate([f for _, f in self._mesh_arrays]) if self._mesh_arrays else np.zeros((0, 3)),
            cam_target=np.stack(self.cam_target), success=np.stack(self.success), time=np.asarray(self.time),
        )
        for name in self.assets:
            arrays[f"pos__{name}"] = np.stack(self.pos[name])  # [T, rows, bodies, 3]
            arrays[f"quat__{name}"] = np.stack(self.quat[name])
        meta = dict(self.meta, **(extra or {}))
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
        return path
