"""rl_games helpers shared by the Homo and Hetero training scripts.

* ``fix_minibatch``: the only PPO change RULES.md allows. The official Factory config uses minibatch 512 with
  horizon 128; with 3 envs one rollout holds only 3 x 128 = 384 samples, so the minibatch has to shrink. The
  rule here: keep the official value when it divides the rollout, otherwise take the largest value below it
  that divides the rollout and is a multiple of the LSTM sequence length. Every change is returned so the
  script can record it.
* ``StepTimer``: a gym wrapper that adds up the time spent inside ``env.step`` (optionally synchronising the
  GPU first, so the time is not charged to the policy's pending kernels).
* ``GpuSampler``: GPU utilisation and memory sampled in a background thread through NVML, if available.
* ``MetricsObserver``: rl_games' ``IsaacAlgoObserver`` (tensorboard logging unchanged) plus one JSON line per
  epoch with env steps, wall-clock time, env-step time, per-task success and return, and GPU usage.
* ``per_task_advantages``: optional per-task advantage normalisation for Hetero (an analysis experiment, not
  part of the official config). rl_games normalises the advantages of the whole rollout at once; with three
  tasks of very different reward scales in one rollout, the task with the smallest rewards early on (nut)
  ends up with tiny advantages. Homo gets per-task normalisation for free (one task per rollout). This
  re-normalises each task's samples separately, right after rl_games built the dataset and before the update.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import gymnasium as gym
import torch

try:
    from rl_games.common.algo_observer import IsaacAlgoObserver
except ImportError:  # offline checks: rl_games stubbed or absent
    IsaacAlgoObserver = object


def fix_minibatch(agent_cfg: dict, num_envs: int) -> list[dict]:
    """Shrink PPO / central-value minibatches that do not divide one rollout. Returns the changes made."""
    cfg = agent_cfg["params"]["config"]
    batch = num_envs * cfg["horizon_length"]
    seq = cfg.get("seq_length", 1) if "rnn" in agent_cfg["params"].get("network", {}) else 1
    changes = []

    def fit(official: int) -> int:
        if official <= batch and batch % official == 0:
            return official
        for m in range(min(official, batch), 0, -1):
            if batch % m == 0 and m % seq == 0:
                return m
        raise ValueError(f"no minibatch size divides batch {batch} with sequence length {seq}")

    for where, d in (("config", cfg), ("config.central_value_config", cfg.get("central_value_config"))):
        if d is None or "minibatch_size" not in d:
            continue
        new = fit(int(d["minibatch_size"]))
        if new != d["minibatch_size"]:
            changes.append(dict(field=f"{where}.minibatch_size", official=d["minibatch_size"], used=new,
                                reason=f"rollout = {num_envs} envs x {cfg['horizon_length']} steps = {batch}"))
            d["minibatch_size"] = new
    return changes


def epochs_for_env_steps(env_steps: int, num_envs: int, horizon: int) -> int:
    """PPO iterations needed to collect at least ``env_steps`` env steps."""
    return math.ceil(env_steps / (num_envs * horizon))


class StepTimer(gym.Wrapper):
    """Adds up wall time spent in env.step; ``pop()`` returns and clears it."""

    def __init__(self, env, sync_cuda: bool = True):
        super().__init__(env)
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.time = 0.0
        self.calls = 0

    def step(self, action):
        if self.sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.env.step(action)
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.time += time.perf_counter() - t0
        self.calls += 1
        return out

    def pop(self) -> tuple[float, int]:
        t, n = self.time, self.calls
        self.time, self.calls = 0.0, 0
        return t, n


class GpuSampler:
    """GPU utilisation (%) and memory used (MiB) of one device, sampled every ``period`` seconds.

    ``device_index`` is the CUDA index as the process sees it. NVML ignores CUDA_VISIBLE_DEVICES, so the
    device is matched by UUID (falling back to the CUDA_VISIBLE_DEVICES list, then to the same index).
    Note that utilisation counts every process on the GPU, so it is only meaningful on an otherwise idle GPU.
    """

    def __init__(self, device_index: int = 0, period: float = 0.5):
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = self._find_handle(pynvml, device_index)
            self.name = pynvml.nvmlDeviceGetName(self._handle)
            self.driver = pynvml.nvmlSystemGetDriverVersion()
            self.ok = True
        except Exception as e:  # noqa: BLE001 - NVML is optional
            self.error = repr(e)
            self.name = torch.cuda.get_device_name(device_index) if torch.cuda.is_available() else None
            self.driver = None
        self.period = period
        self._samples: list[tuple[float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _find_handle(pynvml, device_index: int):
        try:
            uuid = str(torch.cuda.get_device_properties(device_index).uuid)
            return pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
        except Exception:  # noqa: BLE001
            pass
        import os

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            entry = visible.split(",")[device_index].strip()
            if entry.isdigit():
                return pynvml.nvmlDeviceGetHandleByIndex(int(entry))
            return pynvml.nvmlDeviceGetHandleByUUID(entry)
        return pynvml.nvmlDeviceGetHandleByIndex(device_index)

    def _run(self):
        while not self._stop.is_set():
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
                with self._lock:
                    self._samples.append((float(util), float(mem)))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period)

    def pop(self) -> dict:
        """Mean utilisation and max memory since the last call."""
        with self._lock:
            s, self._samples = self._samples, []
        if not s:
            return {}
        return dict(gpu_util_mean=sum(u for u, _ in s) / len(s), gpu_mem_max_mib=max(m for _, m in s), gpu_samples=len(s))

    def close(self):
        self._stop.set()


def per_task_advantages(algo, onehot: slice, task_names, renormalize: bool) -> dict:
    """Per-task statistics of the advantages rl_games just normalised over the whole rollout, and optionally
    re-normalise them per task (each task's samples to mean 0, std 1). The task of a sample is read from the
    task one-hot in its observation, as the gradient probe does. Returns the statistics before re-normalising."""
    d = algo.dataset.values_dict
    adv = d["advantages"]
    task = d["obs"][:, onehot].argmax(-1)
    mask = d.get("rnn_masks")
    valid = (mask.reshape(-1) > 0) if mask is not None else torch.ones_like(task, dtype=torch.bool)
    stats = {}
    with torch.no_grad():
        new = adv.clone()
        for t, name in enumerate(task_names):
            sel = (task == t) & valid
            n = int(sel.sum())
            if n < 2:
                continue
            m, sd = adv[sel].mean(), adv[sel].std()
            stats[f"adv_mean_{name}"] = float(m)
            stats[f"adv_std_{name}"] = float(sd)
            stats[f"adv_abs_mean_{name}"] = float(adv[sel].abs().mean())
            if renormalize:
                rows = task == t
                new[rows] = (adv[rows] - m) / (sd + 1e-8)
        if renormalize:
            adv.copy_(new)  # in place: the actor's dataset and the central value's dataset share this tensor
    return stats


def pooled_task_advantages(algo, raw: torch.Tensor, onehot: slice, task_names, state: dict) -> dict:
    """Normalise the advantages with statistics pooled over the most recent samples of every task (2026-10-07
    meeting, TODO1: a fair counterpart of rl_games' whole-rollout normalisation for Homo).

    ``raw`` is the advantage before rl_games normalised it (returns - values, captured in prepare_dataset). For each
    task present in this rollout, its (count, mean, variance) replaces the one kept in ``state``; the advantages are
    then normalised with the pooled mean and variance of all tasks in ``state``. In Hetero every rollout holds all
    tasks, so this is rl_games' whole-rollout normalisation; in Homo a rollout holds one task, and the other tasks
    enter through their latest rollouts, so each task is again judged against all tasks together.
    Returns per-task statistics of the advantages the update will use."""
    d = algo.dataset.values_dict
    adv = d["advantages"]
    task = d["obs"][:, onehot].argmax(-1)
    mask = d.get("rnn_masks")
    valid = (mask.reshape(-1) > 0) if mask is not None else torch.ones_like(task, dtype=torch.bool)
    stats = {}
    with torch.no_grad():
        raw = raw.reshape(-1).to(adv.device)
        for t, name in enumerate(task_names):
            sel = (task == t) & valid
            n = int(sel.sum())
            if n >= 2:
                state[name] = (n, float(raw[sel].mean()), float(raw[sel].var(unbiased=False)))
        total = sum(n for n, _, _ in state.values())
        mean = sum(n * m for n, m, _ in state.values()) / total
        var = sum(n * (v + (m - mean) ** 2) for n, m, v in state.values()) / total
        new = (raw - mean) / (var ** 0.5 + 1e-8)
        adv.copy_(new)  # in place: the actor's dataset and the central value's dataset share this tensor
        stats["adv_pool_mean"], stats["adv_pool_std"], stats["adv_pool_tasks"] = mean, var ** 0.5, len(state)
        for t, name in enumerate(task_names):
            sel = (task == t) & valid
            if int(sel.sum()) >= 2:
                stats[f"adv_mean_{name}"] = float(adv[sel].mean())
                stats[f"adv_std_{name}"] = float(adv[sel].std())
                stats[f"adv_abs_mean_{name}"] = float(adv[sel].abs().mean())
    return stats


class MetricsObserver(IsaacAlgoObserver):
    """IsaacAlgoObserver plus one JSON line per PPO epoch (``metrics.jsonl``)."""

    TASK_KEYS = ("successes_", "return_", "rew_")

    def __init__(self, out_dir, task_names, step_timer=None, gpu=None, warmup_epochs: int = 1, env=None, probe=None,
                 epoch_end_hook=None, adv_onehot: slice | None = None, adv_per_task: bool = False,
                 adv_mode: str | None = None):
        if IsaacAlgoObserver is not object:
            super().__init__()
        self.path = Path(out_dir) / "metrics.jsonl"
        self.episodes_path = Path(out_dir) / "episodes.jsonl"
        self.epoch_end_hook = epoch_end_hook  # called once per epoch with the epoch number (homo switching)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_names = tuple(task_names)
        self.step_timer = step_timer
        self.gpu = gpu
        self.warmup_epochs = warmup_epochs
        self.env = env  # the unwrapped HeteroFactoryEnv, for per-epoch task counts
        self.probe = probe  # grad_probe.GradientProbe or None
        # per-task advantage statistics (and, if adv_per_task, re-normalisation); needs the one-hot's columns
        self.adv_onehot = adv_onehot
        self.adv_mode = adv_mode or ("per_task" if adv_per_task else "global")
        self.adv_per_task = self.adv_mode == "per_task"
        self.adv_stats = {}
        self._adv_pool = {}  # pooled mode: task -> (count, mean, variance) of its latest raw advantages
        self._raw_adv = None
        self.latest = {}
        self.t_start = None
        self.t_warm = None
        self.last_frame = 0

    def after_init(self, algo):
        if IsaacAlgoObserver is not object:
            super().after_init(algo)
        self.algo = algo
        if self.adv_mode == "pooled":
            # keep the advantages from before rl_games normalises them (same formula as its prepare_dataset)
            orig = algo.prepare_dataset

            def prepare_dataset(batch_dict, _orig=orig):
                self._raw_adv = torch.sum(batch_dict["returns"] - batch_dict["values"], dim=1).detach().clone()
                return _orig(batch_dict)

            algo.prepare_dataset = prepare_dataset
        self.t_start = time.perf_counter()
        self.t_last = self.t_start

    def after_steps(self):
        """rl_games calls this after building the PPO dataset, before the first optimisation step."""
        if IsaacAlgoObserver is not object:
            super().after_steps()
        if self.adv_onehot is not None and self.adv_mode == "pooled":
            self.adv_stats = pooled_task_advantages(self.algo, self._raw_adv, self.adv_onehot, self.task_names,
                                                    self._adv_pool)
        elif self.adv_onehot is not None:
            self.adv_stats = per_task_advantages(self.algo, self.adv_onehot, self.task_names, self.adv_per_task)
        if self.probe is not None:
            self.probe.on_update_start(self.algo)  # sees the advantages the update will use

    def process_infos(self, infos, done_indices):
        if IsaacAlgoObserver is not object:
            super().process_infos(infos, done_indices)
        for k, v in infos.items():
            if k.startswith(self.TASK_KEYS) or k in ("successes", "reset_ik_attempts", "homo_task"):
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    self.latest[k] = float(v)
                elif isinstance(v, (int, float)):
                    self.latest[k] = float(v)

    def after_print_stats(self, frame, epoch_num, total_time):
        if IsaacAlgoObserver is not object:
            super().after_print_stats(frame, epoch_num, total_time)
        # The rl_games fork Isaac Lab installs calls this twice per epoch (inside write_stats and right after):
        # write one line per epoch.
        if epoch_num == getattr(self, "_logged_epoch", None):
            return
        self._logged_epoch = epoch_num
        now = time.perf_counter()
        if epoch_num == self.warmup_epochs:
            self.t_warm = now
        rec = dict(
            epoch=int(epoch_num),
            env_steps=int(frame),
            epoch_env_steps=int(frame - self.last_frame),
            wall_time=now - self.t_start,
            wall_time_after_warmup=(now - self.t_warm) if self.t_warm is not None else None,
            epoch_time=now - self.t_last,
            rl_games_total_time=float(total_time),
        )
        if self.step_timer is not None:
            env_time, env_calls = self.step_timer.pop()
            rec.update(env_step_time=env_time, env_step_calls=env_calls)
        if self.gpu is not None:
            rec.update(self.gpu.pop())
        if self.env is not None:
            # samples collected per task during this epoch's rollout, and the env split at the epoch's end
            if hasattr(self.env, "pop_samples_per_task"):
                rec["samples_per_task"] = dict(zip(self.task_names, self.env.pop_samples_per_task()))
            counts = torch.bincount(self.env.task_ids, minlength=len(self.task_names)).tolist()
            rec["envs_per_task_at_end"] = dict(zip(self.task_names, counts))
        if self.probe is not None:
            # gradient diagnostics are not part of training: report wall time with them taken out
            rec.update(self.probe.last_summary)
            rec["grad_diag_time_total"] = self.probe.time_total
            rec["wall_time_excl_diag"] = rec["wall_time"] - self.probe.time_total
            self.probe.last_summary = {}
        rec.update(self.adv_stats)
        rec.update(self.latest)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        # one line per task per episode that ended during this epoch, stamped with this epoch's steps and time
        log = getattr(self.env, "episode_log", None) if self.env is not None else None
        if log:
            with open(self.episodes_path, "a", encoding="utf-8") as f:
                for e in log:
                    f.write(json.dumps(dict(e, epoch=int(epoch_num), env_steps=int(frame), wall_time=rec["wall_time"],
                                            wall_time_excl_diag=rec.get("wall_time_excl_diag"))) + "\n")
            log.clear()
        self.last_frame = frame
        self.t_last = now
        if self.epoch_end_hook is not None:
            self.epoch_end_hook(int(epoch_num))


def make_homo_rollout_hook(env, block: int, horizon: int, probe=None):
    """Homo switching at rollout boundaries: every ``block`` PPO iterations, the next task starts.

    rl_games runs exactly ``horizon`` env steps per iteration and then calls the observer's
    ``after_print_stats``, which calls the returned hook with the iteration number. At the end of iteration
    ``e``, if iteration ``e + 1`` closes a block, the env is armed to time out on the last step of ``e + 1``,
    so the reset puts the next task in place before iteration ``e + 2`` collects anything: every rollout holds
    one task. With ``block == 1`` the first block closes with iteration 1, so the env is armed right away
    (before training starts). The gradient probe is asked for the two updates around each switch.
    """
    if block < 1:
        raise ValueError("block must be >= 1")
    if block == 1:
        env.schedule_homo_switch(after_steps=horizon)

    def hook(epoch: int):
        if (epoch + 1) % block == 0:
            env.schedule_homo_switch(after_steps=horizon)
            if probe is not None:
                probe.force_next(2)  # the last update of this task and the first of the next

    return hook


def software_versions() -> dict:
    """Versions to record with every run (RULES.md: same software commit/version)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for pkg in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaacsim", "rl_games", "rl-games", "gymnasium"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out
