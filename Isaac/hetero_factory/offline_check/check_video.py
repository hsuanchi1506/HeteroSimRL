"""Offline checks for the video script: frame composition, the writer, and where the cameras are created.

1. compose(): one row per task, tiles in the right place, green border only on successful envs, label strip,
   empty tiles padded.
2. Writer: PNG fallback always; mp4 if imageio-ffmpeg is installed.
3. The scene hook the video script uses (_setup_extra_scene) runs before the union scene clones env 0, as in
   Isaac Lab's official camera tasks, and once in the big-list scene (which does not clone).
4. record_video.py compiles.
"""
import py_compile
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "stubs"), str(HERE.parents[1])]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hetero_factory import video_utils as vu  # noqa: E402

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- 1. compose
W, H, N = 40, 30, 12
frames = np.zeros((N, H, W, 4), dtype=np.uint8)
for e in range(N):
    frames[e, ..., :3] = 10 * e  # each env a flat grey level
tiles = [[(0, "peg_insert"), (3, "peg_insert"), (6, "peg_insert")],
         [(1, "gear_mesh"), (4, "gear_mesh")],
         [(2, "nut_thread"), (5, "nut_thread"), (8, "nut_thread")]]
success = [e in (3, 5) for e in range(N)]
img = vu.compose(frames, tiles, success, 1.5, (W, H), per_row=3)
check(img.shape == (3 * H, 3 * W, 3), f"grid shape {img.shape}")
check(tuple(img[H + H // 2, W + W // 2]) == (40, 40, 40), "env 4 in row 2, column 2 (centre pixel = its grey)")
check(tuple(img[H + H // 2, 2 * W + W // 2]) == (0, 0, 0), "missing tile padded black")
check(tuple(img[H // 2, W + 1]) == vu.SUCCESS_BORDER, "env 3 succeeded: green border")
check(tuple(img[H // 2, 1]) == vu.PLAIN_BORDER, "env 0 did not: grey border")
check(tuple(img[2 * H + H // 2, W + 1]) == vu.SUCCESS_BORDER, "env 5 (nut) succeeded: green border")
has_pil = True
try:
    import PIL  # noqa: F401
except ImportError:
    has_pil = False
if has_pil:
    strip = img[5, 30:W - 4]
    check(int(strip.max()) > 0 and tuple(img[13, W - 6]) == (0, 0, 0), "label strip drawn at the top of each tile")

# ---- 2. writer
with tempfile.TemporaryDirectory() as d:
    w = vu.Writer(Path(d) / "v", fps=15, force_png=True)
    for _ in range(3):
        w.add(img)
    msg = w.close()
    check(len(list((Path(d) / "v").glob("*.png"))) == 3 and "ffmpeg -framerate 15" in msg, "PNG fallback + ffmpeg hint")
    try:
        import imageio_ffmpeg  # noqa: F401

        w = vu.Writer(Path(d) / "m", fps=15)
        for _ in range(3):
            w.add(np.ascontiguousarray(img[: H * 2, : W * 2]))
        w.close()
        check(w.kind == "mp4" and (Path(d) / "m.mp4").stat().st_size > 0, "mp4 written with imageio-ffmpeg")
    except ImportError:
        print("skip mp4: imageio-ffmpeg not installed here")

# ---- 3. camera hook order
from isaaclab._fake_runtime import STATE  # noqa: E402

from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.hetero_factory_env_cfg import HeteroFactoryEnvCfg, HeteroFactoryListEnvCfg  # noqa: E402
from hetero_factory.hetero_factory_list_env import HeteroFactoryListEnv  # noqa: E402

calls = []


def recorder(base):
    class Rec(base):
        def _setup_extra_scene(self):
            super()._setup_extra_scene()
            calls.append((base.__name__, getattr(self.scene, "cloned", False), len(STATE["spawned"])))

    return Rec


for base, cfg_cls in ((HeteroFactoryEnv, HeteroFactoryEnvCfg), (HeteroFactoryListEnv, HeteroFactoryListEnvCfg)):
    cfg = cfg_cls()
    cfg.scene = cfg.scene.replace(num_envs=6)
    calls.clear()
    recorder(base)(cfg)
    check(len(calls) == 1 and calls[0][1] is False,
          f"{base.__name__}: extra-prims hook runs once, before any cloning {calls}")

# ---- 4. the script compiles
py_compile.compile(str(HERE.parents[1] / "scripts" / "record_video.py"), doraise=True)
check(True, "record_video.py compiles")

print(f"\n{len(fails)} failures")
sys.exit(1 if fails else 0)
