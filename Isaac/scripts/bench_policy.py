"""Step time over whole episodes with a trained policy (or random actions), Hetero vs Homo (RULES section 6).

bench_scene.py times steps 20-140 of an episode with random actions. In training the step time grows along the
episode (parts in contact) and every episode end runs the official reset (IK to the start pose, grasp) inside
env.step. This runs whole episodes with a trained checkpoint's actions (sampled, as in training, unless
--deterministic) and times every env.step the way training's StepTimer does (synchronised before and after), so
the mean over an episode, the reset step and the time by episode phase can be compared with the training logs.
Policy inference is not timed (training does not count it in env_step_time either).

RULES section 6 additions:
  --contacts         after every step (outside the timed region) read from PhysX the contact points and force between
                     each env's active held part and its fixed part ("task contact") and between the held part and
                     the two fingers ("grasp contact"), and the official "engaged" check of the Factory reward. Each
                     env is then in one phase: pre-contact (held part not touching the fixed part), contact
                     (touching, not engaged) or heavy-contact / insertion (engaged). Stored per step and task.
                     The official Factory config already enables contact reporting on the robot and every part.
  --profile_windows  torch.profiler (CUPTI) windows of --profile_steps policy steps starting at these step indices;
                     each env.step is profiled on its own (policy inference between steps is not recorded): kernel
                     time by group (narrowphase = contact, solver, articulation), launches, and how much of the step
                     the GPU was busy (the rest is the GPU waiting for the CPU: launches, synchronisation).
                     Profiled steps are flagged and left out of the timing summary.

Several processes on one GPU (2026-10-07 meeting, "split the compute"): with --barrier DIR --barrier_n K every process
writes DIR/ready_<pid> after its reset and waits until K processes are ready, so K single-task processes start
stepping together. Each step also records its wall-clock end time (time.time()), so the overlap window can be found.

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/bench_policy.py --headless --num_envs 128 \\
        --checkpoint logs/rl_games/Factory/hetero_n128_s1_ptadv/nn --homo_task gear_mesh --out runs/x.json

``--checkpoint random`` uses uniform random actions in [-1, 1] instead of a policy.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

TASKS = ("peg_insert", "gear_mesh", "nut_thread")
parser = argparse.ArgumentParser(description="Per-step time over whole episodes with a trained policy.")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="A .pth file, a run's nn/ directory (last checkpoint), or 'random'.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--homo_task", choices=TASKS, default=None, help="Every env on this task (default: hetero split).")
parser.add_argument("--episodes", type=int, default=2, help="Whole episodes to run (each ends with a reset step).")
parser.add_argument("--deterministic", action="store_true", help="The policy's mean action instead of sampling.")
parser.add_argument("--slow_active_pose", action="store_true",
                    help="Read the active parts one task at a time (the code the training runs used).")
parser.add_argument("--phase_bin", type=int, default=64, help="Episode-phase bin width (policy steps) in the summary.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many steps even without a reset.")
parser.add_argument("--contacts", action="store_true", help="Record contacts and contact phases (see above).")
parser.add_argument("--profile_windows", type=str, default="", help="Comma-separated step indices, e.g. 5,300.")
parser.add_argument("--profile_steps", type=int, default=3, help="Policy steps per profiling window.")
parser.add_argument("--task_counts", type=str, default=None,
                    help="Hetero with fixed envs per task, e.g. 128,43,43 (peg,gear,nut; sets --num_envs to the sum).")
parser.add_argument("--barrier", type=str, default=None, help="Directory for the start barrier (see above).")
parser.add_argument("--barrier_n", type=int, default=1, help="Processes that must be ready before stepping.")
parser.add_argument("--collision_stack_size", type=int, default=None,
                    help="PhysX GPU collision stack in bytes (the official 2**28 only fits 128 envs).")
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--out", type=str, required=True, help="JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401
from hetero_factory.hetero_factory_env import HeteroFactoryEnv  # noqa: E402
from hetero_factory.hetero_factory_env_cfg import TASK_SLOTS  # noqa: E402
from hetero_factory.profiling import summarize_window  # noqa: E402
from hetero_factory.rl_utils import GpuSampler, software_versions  # noqa: E402
from hetero_factory.train_common import TASK  # noqa: E402

def resolve_checkpoint(path: str) -> Path:
    """Same rule as eval_policy.py (not imported: that script launches the app when imported)."""
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        cands = [c for c in p.glob("last_*_ep_*.pth") if re.search(r"_ep_(\d+)", c.name)]
        if not cands:
            raise SystemExit(f"no last_*_ep_*.pth in {p}")
        p = max(cands, key=lambda c: int(re.search(r"_ep_(\d+)", c.name).group(1)))
    return p


class ContactProbe:
    """Contacts of each env's active held part, read from PhysX after a policy step (its last physics substep).

    One RigidContactView per task: sensors = that task's held part in every env, filters = that task's fixed part
    and the two finger links of the same env (PhysX pairs the i-th sensor with the i-th match of each filter
    pattern, as Isaac Lab's ContactSensor relies on). In the union scene every env holds every part, so the row of
    another task describes that env's parked parts; only the row of the env's own task is used.
    """

    FIELDS = ("task_pts_mean", "task_pts_max", "grasp_pts_mean", "task_force_mean", "pre", "contact", "heavy")

    def __init__(self, het, max_per_pair: int = 256):
        self.het = het
        sim_view = het._robot._physics_sim_view
        links = het._robot.root_physx_view.link_paths[0]
        fingers = [next(p for p in links if p.endswith("/" + n)) for n in ("panda_leftfinger", "panda_rightfinger")]

        def glob(path):
            return re.sub(r"/env_\d+/", "/env_*/", path)

        def body(slot):
            paths = het._slots[slot].root_physx_view.link_paths[0]
            assert len(paths) == 1, (slot, paths)
            return paths[0]

        self.views, self.rows, self.paths = [], [], {}
        for name in het.task_names:
            held, fixed = body(TASK_SLOTS[name]["held"]), body(TASK_SLOTS[name]["fixed"])
            view = sim_view.create_rigid_contact_view(
                glob(held), filter_patterns=[glob(fixed)] + [glob(f) for f in fingers],
                max_contact_data_count=max_per_pair * 3 * het.num_envs)
            env_of_row = [int(re.search(r"/env_(\d+)/", p).group(1)) for p in view.sensor_paths]
            rows = torch.empty(het.num_envs, dtype=torch.long, device=het.device)
            rows[torch.tensor(env_of_row, device=het.device)] = torch.arange(len(env_of_row), device=het.device)
            self.views.append(view)
            self.rows.append(rows)
            self.paths[name] = dict(held=glob(held), fixed=glob(fixed), fingers=[glob(f) for f in fingers])

    def read(self) -> dict:
        het, n, dev = self.het, self.het.num_envs, self.het.device
        task_pts = torch.zeros(n, device=dev)
        grasp_pts = torch.zeros(n, device=dev)
        task_force = torch.zeros(n, device=dev)
        for t, (view, rows) in enumerate(zip(self.views, self.rows)):
            ids = torch.nonzero(het.task_ids == t, as_tuple=False).reshape(-1)
            if not ids.numel():
                continue
            counts = view.get_contact_data(dt=het.physics_dt)[4].reshape(view.sensor_count, view.filter_count).float()
            forces = view.get_contact_force_matrix(dt=het.physics_dt).reshape(view.sensor_count, view.filter_count, 3)
            r = rows[ids]
            task_pts[ids] = counts[r, 0].to(dev)
            grasp_pts[ids] = (counts[r, 1] + counts[r, 2]).to(dev)
            task_force[ids] = torch.linalg.vector_norm(forces[r, 0], dim=-1).to(dev)
        engaged = het._get_curr_successes(height_threshold=het.tp.engage_height_threshold,
                                          check_rot=torch.zeros(n, dtype=torch.bool, device=dev))
        heavy = engaged
        contact = ~engaged & (task_pts > 0)
        pre = ~engaged & (task_pts == 0)
        out = {}
        for t, name in enumerate(het.task_names):
            m = het.task_ids == t
            if not m.any():
                continue
            vals = torch.stack([task_pts[m].mean(), task_pts[m].max(), grasp_pts[m].mean(), task_force[m].mean(),
                                pre[m].float().mean(), contact[m].float().mean(), heavy[m].float().mean()])
            out[name] = [round(v, 4) for v in vals.tolist()]
        return out


SCALARS = ("wall_ms_per_policy_step_profiled", "kernel_launches", "launches_per_physx_step",
           "kernel_time_ms_per_policy_step", "gpu_span_ms_per_policy_step", "gpu_busy_fraction_of_span",
           "mean_kernel_us", "gpu_busy_ms_per_policy_step", "gpu_busy_fraction_of_step")


def merge_steps(parts: list) -> dict:
    """Average the one-step summaries of a profiling window."""
    n = len(parts)
    for q in parts:
        busy = (q["gpu_busy_fraction_of_span"] or 0.0) * q["gpu_span_ms_per_policy_step"]
        q["gpu_busy_ms_per_policy_step"] = busy
        q["gpu_busy_fraction_of_step"] = busy / q["wall_ms_per_policy_step_profiled"]
    out = dict(policy_steps=n, physx_steps=sum(q["physx_steps"] for q in parts))
    for k in SCALARS:
        vals = [q[k] for q in parts if q.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    groups = {}
    for q in parts:
        for g, v in q["groups"].items():
            acc = groups.setdefault(g, dict(launches_per_physx_step=0.0, ms_per_policy_step=0.0))
            acc["launches_per_physx_step"] += v["launches_per_physx_step"] / n
            acc["ms_per_policy_step"] += v["ms_per_policy_step"] / n
    total = sum(v["ms_per_policy_step"] for v in groups.values())
    for v in groups.values():
        v["share"] = v["ms_per_policy_step"] / total if total else None
    out["groups"] = dict(sorted(groups.items(), key=lambda kv: -kv[1]["ms_per_policy_step"]))
    kern = {}
    for q in parts:
        for k in q["top_kernels"]:
            a = kern.setdefault(k["name"], dict(name=k["name"], group=k["group"], ms_per_policy_step=0.0,
                                                launches_per_physx_step=0.0))
            a["ms_per_policy_step"] += k["ms_per_policy_step"] / n
            a["launches_per_physx_step"] += k["launches_per_physx_step"] / n
    out["top_kernels"] = sorted(kern.values(), key=lambda a: -a["ms_per_policy_step"])[:40]
    return out


def other_gpu_processes():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByUUID(
            (lambda s: s if s.startswith("GPU-") else f"GPU-{s}")(str(torch.cuda.get_device_properties(0).uuid)))
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h) + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
        return sorted({(p.pid, round((p.usedGpuMemory or 0) / 2**20)) for p in procs if p.pid != os.getpid()})
    except Exception as e:  # noqa: BLE001
        return [repr(e)]


def main():
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.assignment.mode = "hetero"
    if args_cli.homo_task is not None:
        env_cfg.assignment.mode = "homo"
        env_cfg.assignment.homo_task = TASKS.index(args_cli.homo_task)
    if args_cli.task_counts:
        env_cfg.assignment.counts = tuple(int(c) for c in args_cli.task_counts.split(","))
        env_cfg.scene.num_envs = sum(env_cfg.assignment.counts)
    env_cfg.fast_active_pose = not args_cli.slow_active_pose
    if args_cli.collision_stack_size is not None:
        env_cfg.sim.physx.gpu_collision_stack_size = args_cli.collision_stack_size
    het = HeteroFactoryEnv(env_cfg, render_mode=None)

    pcfg = agent_cfg["params"]
    env = RlGamesVecEnvWrapper(
        het, pcfg["config"]["device"], pcfg["env"].get("clip_observations", math.inf),
        pcfg["env"].get("clip_actions", math.inf), pcfg["env"].get("obs_groups"),
        pcfg["env"].get("concate_obs_groups", True),
    )
    agent = None
    ckpt = args_cli.checkpoint
    if ckpt != "random":
        vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
        ckpt = str(resolve_checkpoint(ckpt))
        pcfg["load_checkpoint"] = True
        pcfg["load_path"] = ckpt
        pcfg["config"]["num_actors"] = het.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(ckpt)
        agent.reset()
    gen = torch.Generator(device=het.device).manual_seed(args_cli.seed)
    shape = (het.num_envs, het.cfg.action_space)

    rep = dict(checkpoint=ckpt, deterministic=args_cli.deterministic, homo_task=args_cli.homo_task,
               num_envs=het.num_envs, fast_active_pose=env_cfg.fast_active_pose, seed=args_cli.seed,
               task_counts=torch.bincount(het.task_ids, minlength=len(TASKS)).tolist(),
               max_episode_length=int(het.max_episode_length), gpu=torch.cuda.get_device_name(0),
               versions=software_versions(), other_gpu_processes_start=other_gpu_processes(),
               gpu_collision_stack_size=env_cfg.sim.physx.gpu_collision_stack_size, decimation=het.cfg.decimation)
    probe = ContactProbe(het) if args_cli.contacts else None
    if probe is not None:
        rep["contact_fields"] = ContactProbe.FIELDS
        rep["contact_paths"] = probe.paths
    windows = sorted(int(w) for w in args_cli.profile_windows.split(",") if w.strip())
    in_window = {w + k: w for w in windows for k in range(args_cli.profile_steps)}
    profiles, window_parts = [], []
    # steps: [phase before the step, ms, 1 if the step ran a reset, 1 if profiled, contacts after the step or None]
    steps, records = [], []
    gpu = GpuSampler(0, period=0.5)
    gpu.pop()
    with torch.inference_mode():  # as in eval_policy.py: env buffers become inference tensors during the steps
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        if agent is not None:
            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()
        het.episode_log.clear()
        if args_cli.barrier:
            bdir = Path(args_cli.barrier)
            bdir.mkdir(parents=True, exist_ok=True)
            torch.cuda.synchronize()
            (bdir / f"ready_{os.getpid()}").write_text(str(time.time()))
            while len(list(bdir.glob("ready_*"))) < args_cli.barrier_n:
                time.sleep(0.05)
            rep["barrier_passed_at"] = time.time()
        rep["stepping_started_at"] = time.time()
        n_resets = 0
        while n_resets < args_cli.episodes and (args_cli.max_steps is None or len(steps) < args_cli.max_steps):
            i = len(steps)
            profiled = i in in_window
            if agent is None:
                actions = torch.rand(shape, device=het.device, generator=gen) * 2 - 1
            else:
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=args_cli.deterministic)
            phase = int(het.episode_length_buf[0])  # all envs share the episode clock (they reset together)
            torch.cuda.synchronize()
            if profiled:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
            t0 = time.perf_counter()
            obs, _, dones, _ = env.step(actions)
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - t0)
            if profiled:
                prof.__exit__(None, None, None)
                window_parts.append(summarize_window(prof, 1, het.cfg.decimation, ms / 1000))
                del prof
            if isinstance(obs, dict):
                obs = obs["obs"]
            reset = bool(dones.any())
            n_resets += reset
            last_of_window = profiled and in_window.get(i + 1) != in_window[i]
            if last_of_window:
                profiles.append(dict(start_step=in_window[i], episode_phase_end=phase, **merge_steps(window_parts)))
                window_parts = []
            contacts = probe.read() if probe is not None else None  # outside the timed and profiled region
            if last_of_window and contacts is not None:
                profiles[-1]["contacts_after"] = contacts
            steps.append([phase, round(ms, 3), int(reset), int(profiled), contacts, round(time.time(), 4)])
            if agent is not None and agent.is_rnn and agent.states is not None and reset:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            records += [dict(task=e["task"], success=e["success"]) for e in het.episode_log if not e["truncated"]]
            het.episode_log.clear()
    rep["gpu"] = gpu.pop()
    gpu.close()
    rep["other_gpu_processes_end"] = other_gpu_processes()
    rep["profiles"] = profiles

    episodes, cur = [], []
    for s in steps:
        cur.append(s)
        if s[2]:
            episodes.append(cur)
            cur = []
    summ = []
    for ep in episodes:
        plain = [s[1] for s in ep if not s[2] and not s[3]]
        bins = {}
        for p, ms, r, prf, *_ in ep:
            if not r and not prf:
                bins.setdefault(p // args_cli.phase_bin * args_cli.phase_bin, []).append(ms)
        summ.append(dict(steps=len(ep), mean_ms_incl_reset=statistics.mean(s[1] for s in ep),
                         mean_ms_excl_reset=statistics.mean(plain), median_ms_excl_reset=statistics.median(plain),
                         reset_step_ms=[s[1] for s in ep if s[2]][0],
                         phase_bins_ms={str(k): statistics.mean(v) for k, v in sorted(bins.items())}))
    rep["episodes"] = summ
    rep["success"] = records
    rep["steps"] = steps
    Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args_cli.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("BENCH_POLICY " + json.dumps({k: v for k, v in rep.items() if k not in ("steps", "versions", "profiles")}))
    for pr in profiles:
        print("PROFILE_WINDOW " + json.dumps({k: v for k, v in pr.items() if k != "top_kernels"}))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
