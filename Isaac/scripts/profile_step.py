"""Which GPU kernels a Factory env step runs, and for how long (RULES.md sections 6 and 7).

No Nsight needed: PyTorch's profiler records every CUDA kernel of the process through CUPTI, PhysX's own
kernels included. After a warm-up, a few policy steps (each = `decimation` PhysX steps) are profiled with
random actions. Kernels are grouped by name into PhysX broadphase / narrowphase / solver / articulation /
other PhysX, and torch (controller, observations, rewards). Also reported: kernel launches per PhysX step and
how much of the profiled window the GPU was busy (union of kernel intervals), i.e. the idle gaps between
kernels where the GPU waits for the CPU.

    ./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/profile_step.py --scene union --num_envs 1024 --headless \\
        --out runs/profile/union_n1024

Writes <out>/summary.json, <out>/kernels.md and a gzipped Chrome trace (open in https://ui.perfetto.dev).
Profiling slows the steps down: never use these numbers as step times.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Isaac/

parser = argparse.ArgumentParser(description="CUDA kernel profile of Factory env steps.")
parser.add_argument("--scene", choices=["union", "list", "official_peg", "official_gear", "official_nut"], required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--homo_task", choices=["peg_insert", "gear_mesh", "nut_thread"], default=None)
parser.add_argument("--warmup", type=int, default=10, help="Policy steps before profiling.")
parser.add_argument("--steps", type=int, default=3, help="Profiled policy steps.")
parser.add_argument("--collision_stack_size", type=int, default=None)
parser.add_argument("--no_trace", action="store_true", help="Skip the Chrome trace (it can be large).")
parser.add_argument("--cuda_profiler_range", action="store_true",
                    help="For Nsight: wrap the profiled steps in cudaProfilerStart/Stop instead of using the torch "
                         "profiler (run under nsys profile --capture-range=cudaProfilerApi).")
parser.add_argument("--slow_active_pose", action="store_true", help="Read the active parts one task at a time.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, required=True, help="Output directory.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gzip  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import hetero_factory  # noqa: E402, F401
from hetero_factory.profiling import summarize_window  # noqa: E402
from hetero_factory.train_common import TASK, TASK_LIST  # noqa: E402

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
        if args_cli.homo_task is not None:
            cfg.assignment.mode = "homo"
            cfg.assignment.homo_task = ["peg_insert", "gear_mesh", "nut_thread"].index(args_cli.homo_task)
    if args_cli.collision_stack_size is not None:
        cfg.sim.physx.gpu_collision_stack_size = args_cli.collision_stack_size
    if scene not in OFFICIAL:
        cfg.fast_active_pose = not args_cli.slow_active_pose
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    env.reset()
    gen = torch.Generator(device=u.device).manual_seed(args_cli.seed)
    shape = (u.num_envs, u.cfg.action_space)

    def act():
        return torch.rand(shape, device=u.device, generator=gen) * 2 - 1

    for _ in range(args_cli.warmup):
        env.step(act())
    torch.cuda.synchronize()

    if args_cli.cuda_profiler_range:  # Nsight records only this range
        out = Path(args_cli.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        for _ in range(args_cli.steps):
            env.step(act())
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        torch.cuda.cudart().cudaProfilerStop()
        info = dict(scene=scene, homo_task=args_cli.homo_task, num_envs=n, steps=args_cli.steps,
                    physx_steps=args_cli.steps * u.cfg.decimation, wall_ms_per_policy_step=1000 * wall_s / args_cli.steps,
                    fast_active_pose=getattr(cfg, "fast_active_pose", None))
        (out / "nsys_range.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print("NSYS_RANGE " + json.dumps(info))
        env.close()
        return

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(args_cli.steps):
            env.step(act())
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0

    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)
    physx_steps = args_cli.steps * u.cfg.decimation
    summary = dict(scene=scene, task=task, homo_task=args_cli.homo_task, num_envs=n,
                   profiled_policy_steps=args_cli.steps, **summarize_window(prof, args_cli.steps, physx_steps, wall_s))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [f"# {scene} n{n}" + (f" homo {args_cli.homo_task}" if args_cli.homo_task else ""), "",
             f"- kernel launches per PhysX step: {summary['launches_per_physx_step']:.0f}",
             f"- GPU busy {100 * (summary['gpu_busy_fraction_of_span'] or 0):.0f}% of the profiled window",
             f"- kernel time per policy step: {summary['kernel_time_ms_per_policy_step']:.1f} ms "
             f"(window {summary['gpu_span_ms_per_policy_step']:.1f} ms; profiled wall {summary['wall_ms_per_policy_step_profiled']:.1f} ms)",
             "", "| group | launches / PhysX step | ms / policy step | share |", "|---|---|---|---|"]
    for g, v in summary["groups"].items():
        lines.append(f"| {g} | {v['launches_per_physx_step']:.0f} | {v['ms_per_policy_step']:.1f} | {100 * (v['share'] or 0):.0f}% |")
    lines += ["", "| kernel | group | launches / PhysX step | ms / policy step | mean us |", "|---|---|---|---|---|"]
    for k in summary["top_kernels"][:25]:
        lines.append(f"| `{k['name'][:90]}` | {k['group']} | {k['launches_per_physx_step']:.1f} | "
                     f"{k['ms_per_policy_step']:.2f} | {k['mean_us']:.1f} |")
    (out / "kernels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    if not args_cli.no_trace:
        raw = out / "trace.json"
        prof.export_chrome_trace(str(raw))
        with open(raw, "rb") as fi, gzip.open(str(raw) + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        raw.unlink()
    print("PROFILE_REPORT " + json.dumps({k: v for k, v in summary.items() if k not in ("top_kernels", "groups")}))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
