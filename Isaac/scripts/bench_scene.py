"""Step time of the hetero Factory scenes on PhysX: union scene (every env holds all 8 parts, unused ones
parked) vs big-list scene (every env holds only its task's parts). Same tasks, same assignment, same seed and
actions, so the active parts move identically in both (checked: bit-identical poses); any time difference is
the cost of the padding parts, plus how the scene was built.

Three scenes:
  union        the hetero union scene as the official Factory builds it (env 0 cloned, PhysX replication on)
  union_norep  the same union scene built like the big list (every env parsed by PhysX on its own); separates
               the cost of the padding from the cost of building without replication
  list         the big-list scene (replicate_physics=False, no cloning)
  official_*   the official single-task Factory env (peg / gear / nut), as a reference
With ``--homo_task`` the union scenes put every env on one task (the other tasks' parts all parked), which
separates what each task's parked parts cost.

Per run (one scene, one env count, fresh process):
  build_s       gym.make (stage + PhysX parsing)
  reset_s       two full resets (official Factory reset: gravity off, IK to the start pose, grasp; all envs)
  step_ms       env.step in blocks of --block_steps policy steps, synchronised at block ends only
  breakdown_ms  one more block with every stage of DirectRLEnv.step timed separately (synchronised around each
                call, so it adds a little time): action processing, controller, write to sim, PhysX step,
                read back, dones / rewards / observations
  bodies        rigid bodies in the PhysX scene (all articulations x instances)
  physx_stats   PhysX's own counters over --stats_steps more steps (max and mean): contact pairs, solver
                constraints, and the GPU buffer sizes the step actually needed (``gpu_mem_*``); also right
                after a reset, where the grasp makes the most contacts
  gpu           mean utilisation and memory (NVML, whole GPU) during the timed blocks; this process's memory

The official Factory GPU buffers are sized for 128 envs. At 4096 envs the union scene overflows the collision
stack (PhysX drops contacts and says so in the log), so ``--collision_stack_size`` raises that capacity; it
changes no solver setting, and the value used is recorded.

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/bench_scene.py --scene union --num_envs 128 --headless --out x.json

Timing is only meaningful on an otherwise idle GPU: the script records the other processes on the GPU.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

parser = argparse.ArgumentParser(description="Step-time benchmark: union vs big-list hetero Factory scene.")
parser.add_argument("--scene", choices=["union", "union_norep", "list", "official_peg", "official_gear", "official_nut"],
                    required=True)
parser.add_argument("--layout", choices=["interleaved", "blocked"], default="interleaved",
                    help="Hetero env order: tasks interleaved (peg, gear, nut, peg, ...) or in contiguous blocks.")
parser.add_argument("--homo_task", choices=["peg_insert", "gear_mesh", "nut_thread"], default=None,
                    help="Union scenes only: every env on this task instead of the hetero split.")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--actions", choices=["random", "zero"], default="random")
parser.add_argument("--warmup", type=int, default=20, help="Untimed policy steps after the resets.")
parser.add_argument("--blocks", type=int, default=3, help="Timed blocks.")
parser.add_argument("--block_steps", type=int, default=40, help="Policy steps per timed block.")
parser.add_argument("--breakdown_steps", type=int, default=30, help="Policy steps of the per-stage breakdown.")
parser.add_argument("--stats_steps", type=int, default=10, help="Policy steps sampled for PhysX statistics.")
parser.add_argument("--slow_active_pose", action="store_true",
                    help="Read the active parts one task at a time (the code before 2026-10-06), for comparison.")
parser.add_argument("--collision_stack_size", type=int, default=None,
                    help="PhysX GPU collision stack in bytes (default: the official 2**28).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None, help="Write the report as JSON here.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401
from hetero_factory.rl_utils import GpuSampler, software_versions  # noqa: E402
from hetero_factory.train_common import TASK, TASK_LIST  # noqa: E402

# stages of DirectRLEnv.step, in call order; per policy step the middle four run `decimation` times
STAGES = [
    ("env", "_pre_physics_step"),
    ("env", "_apply_action"),
    ("scene", "write_data_to_sim"),
    ("sim", "step"),
    ("scene", "update"),
    ("env", "_get_dones"),
    ("env", "_get_rewards"),
    ("env", "_get_observations"),
]


def sync():
    torch.cuda.synchronize()


def instrument(u, totals):
    """Wrap every stage with synchronised timers; returns a function that removes the wrappers."""
    owners = {"env": u, "scene": u.scene, "sim": u.sim}
    undo = []
    for owner_name, attr in STAGES:
        owner = owners[owner_name]
        orig = getattr(owner, attr)
        key = f"{owner_name}.{attr}"

        def timed(*a, _orig=orig, _key=key, **k):
            sync()
            t = time.perf_counter()
            r = _orig(*a, **k)
            sync()
            totals[_key] += time.perf_counter() - t
            return r

        setattr(owner, attr, timed)  # instance attribute shadows the bound method
        undo.append((owner, attr))

    def remove():
        for owner, attr in undo:
            delattr(owner, attr)

    return remove


def other_gpu_processes():
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByUUID(
            (lambda s: s if s.startswith("GPU-") else f"GPU-{s}")(str(torch.cuda.get_device_properties(0).uuid)))
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h) + pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
        mine = os.getpid()
        own = max((p.usedGpuMemory or 0 for p in procs if p.pid == mine), default=0) / 2**20
        others = sorted({(p.pid, round((p.usedGpuMemory or 0) / 2**20)) for p in procs if p.pid != mine})
        return own, [dict(pid=pid, mem_mib=m) for pid, m in others]
    except Exception as e:  # noqa: BLE001
        return None, [dict(error=repr(e))]


def physx_stats(u):
    """PhysX's statistics of the last simulation step (contact pairs, constraints, GPU buffer needs)."""
    try:
        import omni.usd
        from omni.physx import get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools

        stats = PhysicsSceneStats()
        ok = get_physx_statistics_interface().get_physx_scene_statistics(
            omni.usd.get_context().get_stage_id(), PhysicsSchemaTools.sdfPathToInt(u.cfg.sim.physics_prim_path), stats
        )
        if not ok:
            return {"error": "get_physx_scene_statistics returned False"}
        return {k: int(getattr(stats, k)) for k in dir(stats)
                if not k.startswith("_") and isinstance(getattr(stats, k), int)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def summarize_stats(samples):
    good = [s for s in samples if "error" not in s]
    if not good:
        return samples[:1]
    return {k: dict(max=max(s[k] for s in good), mean=sum(s[k] for s in good) / len(good)) for k in good[0]}


OFFICIAL = {"official_peg": "Isaac-Factory-PegInsert-Direct-v0", "official_gear": "Isaac-Factory-GearMesh-Direct-v0",
            "official_nut": "Isaac-Factory-NutThread-Direct-v0"}


def main():
    scene, n = args_cli.scene, args_cli.num_envs
    task = OFFICIAL.get(scene) or (TASK_LIST if scene == "list" else TASK)
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args_cli.seed
    if scene not in OFFICIAL:
        cfg.assignment.mode = "hetero"
        cfg.assignment.layout = args_cli.layout
        if args_cli.homo_task is not None:
            if scene == "list":
                raise SystemExit("--homo_task needs a union scene (the big list is hetero only)")
            cfg.assignment.mode = "homo"
            cfg.assignment.homo_task = ["peg_insert", "gear_mesh", "nut_thread"].index(args_cli.homo_task)
    if scene == "union_norep":
        cfg.scene.replicate_physics = False
        cfg.scene.clone_in_fabric = False
    if args_cli.collision_stack_size is not None:
        cfg.sim.physx.gpu_collision_stack_size = args_cli.collision_stack_size
    if scene not in OFFICIAL:
        cfg.fast_active_pose = not args_cli.slow_active_pose
    px = cfg.sim.physx
    rep = dict(scene=scene, task=task, homo_task=args_cli.homo_task, layout=args_cli.layout, num_envs=n,
               fast_active_pose=getattr(cfg, "fast_active_pose", None),
               actions=args_cli.actions,
               seed=args_cli.seed,
               replicate_physics=bool(cfg.scene.replicate_physics), clone_in_fabric=bool(cfg.scene.clone_in_fabric),
               gpu=torch.cuda.get_device_name(0), versions=software_versions(),
               decimation=cfg.decimation, physics_dt=cfg.sim.dt,
               solver_position_iterations=px.max_position_iteration_count,
               physx_buffers=dict(gpu_collision_stack_size=px.gpu_collision_stack_size,
                                  gpu_max_rigid_contact_count=px.gpu_max_rigid_contact_count,
                                  gpu_max_rigid_patch_count=px.gpu_max_rigid_patch_count,
                                  gpu_found_lost_pairs_capacity=px.gpu_found_lost_pairs_capacity,
                                  gpu_total_aggregate_pairs_capacity=px.gpu_total_aggregate_pairs_capacity,
                                  gpu_heap_capacity=px.gpu_heap_capacity))

    t0 = time.perf_counter()
    env = gym.make(task, cfg=cfg)
    sync()
    rep["build_s"] = time.perf_counter() - t0
    u = env.unwrapped
    if hasattr(u, "task_ids"):
        rep["task_counts"] = torch.bincount(u.task_ids, minlength=len(u.task_names)).tolist()
    rep["articulations"] = {k: dict(instances=int(a.num_instances), bodies=int(a.num_bodies))
                            for k, a in u.scene.articulations.items()}
    rep["bodies"] = sum(v["instances"] * v["bodies"] for v in rep["articulations"].values())

    resets = []
    for _ in range(2):
        sync()
        t0 = time.perf_counter()
        env.reset()
        sync()
        resets.append(time.perf_counter() - t0)
    rep["reset_s"] = resets
    rep["physx_stats_after_reset"] = physx_stats(u)

    gen = torch.Generator(device=u.device).manual_seed(args_cli.seed)
    shape = (u.num_envs, u.cfg.action_space)

    def act():
        if args_cli.actions == "zero":
            return torch.zeros(shape, device=u.device)
        return torch.rand(shape, device=u.device, generator=gen) * 2 - 1

    for _ in range(args_cli.warmup):
        env.step(act())

    gpu = GpuSampler(0, period=0.2)
    gpu.pop()
    blocks = []
    for _ in range(args_cli.blocks):
        sync()
        t0 = time.perf_counter()
        for _ in range(args_cli.block_steps):
            env.step(act())
        sync()
        blocks.append(1000 * (time.perf_counter() - t0) / args_cli.block_steps)
    rep["step_ms_blocks"] = blocks
    rep["step_ms"] = statistics.median(blocks)
    rep["env_steps_per_s"] = u.num_envs * 1000 / rep["step_ms"]
    rep["gpu"] = gpu.pop()
    gpu.close()
    rep["own_gpu_mem_mib"], rep["other_gpu_processes"] = other_gpu_processes()
    rep["torch_max_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20

    totals = defaultdict(float)
    remove = instrument(u, totals)
    sync()
    t0 = time.perf_counter()
    for _ in range(args_cli.breakdown_steps):
        env.step(act())
    sync()
    wall = time.perf_counter() - t0
    remove()
    k = args_cli.breakdown_steps
    rep["breakdown_ms"] = {key: 1000 * v / k for key, v in totals.items()}
    rep["breakdown_ms"]["total_instrumented"] = 1000 * wall / k
    rep["breakdown_ms"]["other"] = rep["breakdown_ms"]["total_instrumented"] - sum(1000 * v / k for v in totals.values())
    samples = []
    for _ in range(args_cli.stats_steps):
        env.step(act())
        samples.append(physx_stats(u))
    rep["physx_stats"] = summarize_stats(samples)
    rep["episode_steps_used"] = int(u.episode_length_buf.max())  # must stay below the episode length (no reset)
    rep["max_episode_length"] = int(u.max_episode_length)

    print("BENCH_REPORT " + json.dumps(rep))
    if args_cli.out:
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
