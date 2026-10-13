"""Metrics, light checkpoints and Homo switching for ``scripts/train_p2p.py`` (Play2Perfect's vendored rl_games).

``P2PMetricsObserver`` writes one JSON line per PPO epoch to ``metrics.jsonl``:

* epoch counters (rl_games' own, which continue from the resumed checkpoint, and ``epoch_rel`` from 1), env
  steps, wall-clock and env-step time (``StepTimer``), GPU utilisation and memory (``GpuSampler``);
* env steps collected per task, the Homo task;
* episodes that ended during the epoch, summed per task and per SAPG block (``HeteroP2PEnv.pop_episode_stats``):
  count, full successes (every sub-goal reached, the official offline metric), sub-goal ratio, retract
  successes, return, length. Episodes cut by a Homo switch are only counted (``forced``).

Every ``save_every`` epochs it saves a light checkpoint (policy and critic weights with their normalisers) for the
periodic evaluation and videos of RULES section 8. rl_games' own checkpoints are left as they are (with the
released ``save_frequency`` of 3000 and its square-number rule they are rare).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from rl_games.common.algo_observer import AlgoObserver


class P2PMetricsObserver(AlgoObserver):
    def __init__(self, out_dir, env, step_timer=None, gpu=None, save_every: int = 0, epoch_end_hook=None):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "metrics.jsonl"
        self.env = env
        self.step_timer = step_timer
        self.gpu = gpu
        self.save_every = int(save_every)
        self.epoch_end_hook = epoch_end_hook
        self.epoch0 = None
        self.algo = None

    def after_init(self, algo):
        self.algo = algo
        self.t_start = time.perf_counter()
        self.t_last = self.t_start
        self.last_frame = None

    def save_light(self, epoch_rel: int, epoch: int, frame: int):
        a = self.algo
        state = {"model": a.model.state_dict(), "epoch": int(epoch), "epoch_rel": int(epoch_rel), "frame": int(frame)}
        if getattr(a, "has_central_value", False):
            state["assymetric_vf_nets"] = a.central_value_net.state_dict()
        d = self.out_dir / "light"
        d.mkdir(exist_ok=True)
        torch.save(state, d / f"ep{epoch_rel:06d}.pth")

    # The first Homo run (2026-10-12) died when PhysX could not grow a contact buffer (2.2 GB) at 92.6 GB used of
    # 96 GB: from then on every env ended every step. Two guards, neither changes training:
    FREE_GIB_MIN = 8.0  # below this much free GPU memory, hand PyTorch's cached-but-unused blocks back to the driver
    COLLAPSE_EPOCHS = 3  # stop the run after this many epochs in which every env ended every step

    def memory_guard(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        out = dict(gpu_free_gib=free / 2**30, torch_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                   torch_reserved_gib=torch.cuda.memory_reserved() / 2**30)
        if free / 2**30 < self.FREE_GIB_MIN:
            torch.cuda.empty_cache()
            out["empty_cache"] = True
            out["gpu_free_gib_after"] = torch.cuda.mem_get_info()[0] / 2**30
        return out

    def check_collapse(self, rec: dict):
        n = sum(v["all"]["episodes"] + v["all"]["forced"] for v in rec["episodes"].values())
        steps = int(self.env.num_envs) * int(self.algo.horizon_length)
        self._collapsed = getattr(self, "_collapsed", 0) + 1 if n >= 0.9 * steps else 0
        if self._collapsed >= self.COLLAPSE_EPOCHS:
            raise RuntimeError(f"simulation collapsed: {n:.0f} episodes ended in {steps} env steps for "
                               f"{self._collapsed} epochs (see the PhysX errors in the log)")

    def after_print_stats(self, frame, epoch_num, total_time):
        if epoch_num == getattr(self, "_logged_epoch", None):
            return
        self._logged_epoch = epoch_num
        if self.epoch0 is None:
            self.epoch0 = int(epoch_num) - 1
        epoch_rel = int(epoch_num) - self.epoch0
        now = time.perf_counter()
        # rl_games' frame counter continues from the checkpoint and is read before this epoch is added
        env_steps = epoch_rel * int(self.env.num_envs) * int(self.algo.horizon_length)
        rec = dict(epoch=int(epoch_num), epoch_rel=epoch_rel, frame=int(frame), env_steps=env_steps,
                   wall_time=now - self.t_start, epoch_time=now - self.t_last, rl_games_total_time=float(total_time))
        if self.step_timer is not None:
            t, n = self.step_timer.pop()
            rec.update(env_step_time=t, env_step_calls=n)
        if self.gpu is not None:
            rec.update(self.gpu.pop())
        rec["samples_per_task"] = dict(zip(self.env.task_names, self.env.pop_samples_per_task()))
        rec["homo_task"] = self.env.task_names[self.env.homo_task] if self.env.cfg.assignment.mode == "homo" else None
        rec["episodes"] = self.env.pop_episode_stats()
        rec["lr"] = float(getattr(self.algo, "last_lr", float("nan")))
        rec.update(self.memory_guard())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.check_collapse(rec)
        if self.save_every > 0 and epoch_rel % self.save_every == 0:
            self.save_light(epoch_rel, int(epoch_num), int(frame))
        self.t_last = now
        if self.epoch_end_hook is not None:
            self.epoch_end_hook(epoch_rel)


def make_homo_rollout_hook(env, block: int, horizon: int):
    """Homo switching at rollout boundaries (as hetero_factory.rl_utils.make_homo_rollout_hook): at the end of
    relative epoch e, if epoch e + 1 closes a block, every env times out on the last step of epoch e + 1, so the
    next task starts with epoch e + 2 and every rollout holds one task."""
    if block < 1:
        raise ValueError("block must be >= 1")
    if block == 1:
        env.schedule_homo_switch(after_steps=horizon)

    def hook(epoch_rel: int):
        if (epoch_rel + 1) % block == 0:
            env.schedule_homo_switch(after_steps=horizon)

    return hook


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
