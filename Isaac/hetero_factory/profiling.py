"""Kernel-level summary of a torch.profiler window (CUPTI sees every CUDA kernel, PhysX's included).

Shared by scripts/profile_step.py (random actions) and scripts/bench_policy.py (trained-policy windows).
"""

from collections import defaultdict

from torch.autograd import DeviceType

# kernel-name keywords -> group (first match wins); names come from the PhysX 5.6 GPU sources
GROUPS = [
    ("torch", ("at::", "at_cuda", "elementwise", "reduce_kernel", "gemm", "gemv", "cutlass", "cublas", "Kernel<",
               "triton", "vectorized", "unrolled", "index_", "scatter", "gather", "CatArray", "cunn", "fill_")),
    ("physx_broadphase", ("bp", "BP", "Broadphase", "BroadPhase", "aggregate", "Aggregate", "sap", "Sap", "AABB")),
    ("physx_narrowphase", ("triangleTriangle", "Trimesh", "trimesh", "convex", "Convex", "sdf", "Sdf", "SDF", "Np",
                           "midphase", "Midphase", "contact", "Contact", "manifold", "Manifold", "patch", "Patch",
                           "Pair", "pair", "Mesh", "mesh", "Sphere", "Box", "Plane", "Capsule")),
    ("physx_articulation", ("arti", "Arti", "Articulation", "articulation", "Link", "link")),
    ("physx_solver", ("solve", "Solve", "Slab", "slab", "propagate", "Propagate", "integrat", "Integrat",
                      "Constraint", "constraint", "Partition", "partition", "prep", "Prep", "Island", "island",
                      "Body", "body", "TGS", "Velocity", "velocity")),
]


def group_of(name: str) -> str:
    for g, keys in GROUPS:
        if any(k in name for k in keys):
            return g
    return "other"


def union_length(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def summarize_window(prof, policy_steps: int, physx_steps: int, wall_s: float, top: int = 40) -> dict:
    """Kernel launches, kernel time by group, GPU busy fraction of the window (the rest is the GPU waiting for the
    CPU), and the top kernels, per policy step."""
    kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    intervals = [(e.time_range.start, e.time_range.end) for e in kernels]
    span_us = (max(e for _, e in intervals) - min(s for s, _ in intervals)) if intervals else 0.0
    busy_us = union_length(intervals)
    by_name = defaultdict(lambda: [0, 0.0])
    by_group = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        d = e.time_range.end - e.time_range.start
        by_name[e.name][0] += 1
        by_name[e.name][1] += d
        g = group_of(e.name)
        by_group[g][0] += 1
        by_group[g][1] += d
    total_kernel_us = sum(v[1] for v in by_name.values())
    return dict(
        policy_steps=policy_steps, physx_steps=physx_steps, wall_ms_per_policy_step_profiled=1000 * wall_s / policy_steps,
        kernel_launches=len(kernels), launches_per_physx_step=len(kernels) / physx_steps,
        kernel_time_ms_per_policy_step=total_kernel_us / 1000 / policy_steps,
        gpu_span_ms_per_policy_step=span_us / 1000 / policy_steps,
        gpu_busy_fraction_of_span=busy_us / span_us if span_us else None,
        mean_kernel_us=total_kernel_us / len(kernels) if kernels else None,
        groups={g: dict(launches_per_physx_step=c / physx_steps, ms_per_policy_step=t / 1000 / policy_steps,
                        share=t / total_kernel_us if total_kernel_us else None)
                for g, (c, t) in sorted(by_group.items(), key=lambda kv: -kv[1][1])},
        top_kernels=[dict(name=k, group=group_of(k), launches_per_physx_step=c / physx_steps,
                          ms_per_policy_step=t / 1000 / policy_steps, mean_us=t / c)
                     for k, (c, t) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:top]],
    )
