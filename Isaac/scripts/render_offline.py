"""Draw a pose recording of ``record_video.py --renderer offline`` into a tiled mp4 with MuJoCo's OpenGL renderer.

MuJoCo only draws here: every pose comes from the PhysX run that wrote the recording, nothing is simulated.
The model is one mocap body per recorded rigid body, carrying that body's meshes; for each tile the bodies
are moved to that env's poses (env frame) and a free camera looks at the env's fixed-part tip. Parts that do
not exist in an env (big-list scene) are moved out of sight.

Runs outside Isaac Sim (numpy, mujoco, imageio-ffmpeg, pillow), e.g. on gputw1 in ``~/envs/render``:

    MUJOCO_GL=egl ~/envs/render/bin/python scripts/render_offline.py runs/video/union_zero.npz

Writes ``<recording>.mp4`` next to the recording unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/
from hetero_factory.video_utils import Writer, compose  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

GROUND_Z = -1.05  # the Factory scene's ground plane (the table top is at z = 0); Play2Perfect: --ground_z 0
OUT_OF_SIGHT = (0.0, 0.0, -50.0)


def _nums(a, fmt):
    return " ".join(np.char.mod(fmt, np.asarray(a).ravel()))


def load(path: Path):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    return z, meta


def build_model(z, meta, width: int, height: int, ground_z: float = GROUND_Z):
    """MJCF with the recorded meshes; returns (model, {(asset, body): mocap id})."""
    verts, faces = z["mesh_verts"], z["mesh_faces"]
    mesh_xml, static_xml, body_geoms = [], [], {}
    v0 = f0 = 0
    for i, m in enumerate(meta["meshes"]):
        v, f = verts[v0:v0 + m["nv"]], faces[f0:f0 + m["nf"]]
        v0, f0 = v0 + m["nv"], f0 + m["nf"]
        if m["nf"] == 0 or m["nv"] < 4 or np.ptp(v, axis=0).max() < 1e-6:
            continue
        mesh_xml.append(f'<mesh name="m{i}" inertia="shell" vertex="{_nums(v, "%.6g")}" face="{_nums(f, "%d")}"/>')
        rgba = " ".join(f"{c:.3f}" for c in m["rgba"])
        geom = f'<geom type="mesh" mesh="m{i}" rgba="{rgba}"/>'
        if m["body"] is None:
            static_xml.append(geom)
        else:
            body_geoms.setdefault((m["asset"], m["body"]), []).append(geom)
    keys = list(body_geoms)
    bodies_xml = [f'<body name="b{k}" mocap="true">{"".join(body_geoms[key])}</body>' for k, key in enumerate(keys)]
    xml = f"""
<mujoco model="factory_replay">
  <compiler inertiafromgeom="false"/>
  <default><geom contype="0" conaffinity="0" group="1"/></default>
  <visual>
    <global offwidth="{width}" offheight="{height}"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.97" rgb2="0.62 0.66 0.72" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.80 0.80" rgb2="0.72 0.72 0.72" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12"/>
    {"".join(mesh_xml)}
  </asset>
  <worldbody>
    <light pos="0.9 -0.9 2.5" dir="-0.3 0.3 -1" castshadow="true" diffuse="0.55 0.55 0.55"/>
    <geom type="plane" size="4 4 0.1" pos="0 0 {ground_z}" material="grid"/>
    {"".join(static_xml)}
    {"".join(bodies_xml)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    mocap = {key: int(model.body(f"b{k}").mocapid[0]) for k, key in enumerate(keys)}
    return model, mocap


def free_camera(eye_offset):
    """A free camera whose position is ``lookat + eye_offset`` (set lookat per tile)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    e = np.asarray(eye_offset, dtype=float)
    cam.distance = float(np.linalg.norm(e))
    fwd = -e / cam.distance  # MuJoCo: position = lookat - distance * forward
    cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
    cam.elevation = math.degrees(math.asin(fwd[2]))
    return cam


def main():
    parser = argparse.ArgumentParser(description="Draw a Factory pose recording with MuJoCo's renderer.")
    parser.add_argument("recording", type=Path, help="<out>.npz written by record_video.py --renderer offline.")
    parser.add_argument("--out", type=Path, default=None, help="Output path without extension.")
    parser.add_argument("--tile", type=int, nargs=2, default=[320, 240], help="Tile width and height in pixels.")
    parser.add_argument("--eye", type=float, nargs=3, default=[0.25, -0.2, 0.17],
                        help="Camera position relative to the fixed part's tip (m, env frame).")
    parser.add_argument("--per_task", type=int, default=None, help="Tiles per task (default: as recorded).")
    parser.add_argument("--ground_z", type=float, default=GROUND_Z, help="Ground plane height in the env frame.")
    args = parser.parse_args()

    z, meta = load(args.recording)
    w, h = args.tile
    model, mocap = build_model(z, meta, w, h, args.ground_z)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, h, w)
    cam = free_camera(args.eye)

    tiles = [[tuple(cell) for cell in row] for row in meta["tiles"]]
    if args.per_task is not None:
        tiles = [row[: args.per_task] for row in tiles]
    per_row = max(len(row) for row in tiles)
    tile_envs = sorted({e for row in tiles for e, _ in row})

    pos = {a: z[f"pos__{a}"] for a in meta["assets"]}
    quat = {a: z[f"quat__{a}"] for a in meta["assets"]}
    row_of = {a: {e: r for r, e in enumerate(info["rows_env"])} for a, info in meta["assets"].items()}
    body_idx = {a: {b: i for i, b in enumerate(info["body_names"])} for a, info in meta["assets"].items()}
    cam_target, success, times = z["cam_target"], z["success"], z["time"]

    out = args.out or args.recording.with_suffix("")
    writer = Writer(out, 1.0 / meta["dt"])
    frames = np.zeros((meta["num_envs"], h, w, 3), dtype=np.uint8)
    for t in range(len(times)):
        for e in tile_envs:
            for (asset, body), mid in mocap.items():
                r = row_of[asset].get(e)
                if r is None:
                    data.mocap_pos[mid] = OUT_OF_SIGHT
                    continue
                b = body_idx[asset][body]
                data.mocap_pos[mid] = pos[asset][t, r, b]
                data.mocap_quat[mid] = quat[asset][t, r, b]  # w, x, y, z in both Isaac Lab and MuJoCo
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = cam_target[t, e]
            renderer.update_scene(data, camera=cam)
            frames[e] = renderer.render()
        writer.add(compose(frames, tiles, success[t].tolist(), float(times[t]), (w, h), per_row))
    writer.close()
    renderer.close()
    n_mesh = len(meta["meshes"])
    print(f"[INFO] {len(times)} frames, {len(tile_envs)} tiles, {n_mesh} meshes on {len(mocap)} bodies, "
          f"scene={meta.get('scene')} policy={meta.get('policy')}")


if __name__ == "__main__":
    main()
