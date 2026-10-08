"""Frame composition and video writing for scripts/record_video.py (numpy / PIL / imageio only, no Isaac Sim)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TASK_COLORS = {"peg_insert": (80, 160, 255), "gear_mesh": (255, 170, 60), "nut_thread": (190, 110, 255)}
SUCCESS_BORDER = (40, 200, 60)
PLAIN_BORDER = (60, 60, 60)


def label(img: np.ndarray, text: str, color) -> np.ndarray:
    """A black strip with ``text`` at the top of the tile (unchanged if PIL is missing)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=color)
    return np.asarray(im)


def compose(frames: np.ndarray, tiles, success, time_s: float, tile_wh, per_row: int) -> np.ndarray:
    """Grid of camera images, one row per task.

    frames: [num_envs, H, W, C] uint8; tiles: per row a list of (env_id, task name); success: per env bool.
    Each tile gets a 4-pixel border (green while the env meets its success condition) and a label strip.
    """
    w, h = tile_wh
    rows = []
    for row in tiles:
        cells = []
        for env_id, task in row:
            img = np.ascontiguousarray(frames[env_id][..., :3]).copy()
            border = SUCCESS_BORDER if success[env_id] else PLAIN_BORDER
            img[:4], img[-4:], img[:, :4], img[:, -4:] = border, border, border, border
            cells.append(label(img, f"{task}  env {env_id}  t={time_s:4.1f}s", TASK_COLORS.get(task, (255, 255, 255))))
        while len(cells) < per_row:
            cells.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(rows, axis=0)


class Writer:
    """mp4 through imageio (needs imageio-ffmpeg); otherwise PNG frames plus the ffmpeg command to join them."""

    def __init__(self, path, fps: float, force_png: bool = False):
        self.path, self.fps, self.n = Path(path), fps, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.kind = "png"
        if not force_png:
            try:
                import imageio.v2 as imageio

                self.w = imageio.get_writer(str(self.path.with_suffix(".mp4")), fps=fps, codec="libx264", quality=7)
                self.kind = "mp4"
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] no mp4 writer ({e!r}); writing PNG frames")
        if self.kind == "png":
            self.dir = self.path.with_suffix("")
            self.dir.mkdir(parents=True, exist_ok=True)

    def add(self, frame: np.ndarray):
        if self.kind == "mp4":
            self.w.append_data(frame)
        else:
            from PIL import Image

            Image.fromarray(frame).save(self.dir / f"{self.n:05d}.png")
        self.n += 1

    def close(self) -> str:
        if self.kind == "mp4":
            self.w.close()
            msg = f"wrote {self.path.with_suffix('.mp4')} ({self.n} frames)"
        else:
            msg = (f"wrote {self.n} frames to {self.dir}; join with: "
                   f"ffmpeg -framerate {self.fps:g} -i {self.dir}/%05d.png -pix_fmt yuv420p {self.path}.mp4")
        print(f"[INFO] {msg}")
        return msg
