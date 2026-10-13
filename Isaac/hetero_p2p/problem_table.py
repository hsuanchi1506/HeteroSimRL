"""Per-task parameters of the four Play2Perfect assembly problems, read from the official problem registry.

The official env (``isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py``, ``_configure_problem``) holds
one problem per process: the object and receptive URDFs, the object's grasp-box scale (an observation), the
hole's height above the table, and the insertion sub-goal poses in the receptive frame. Here every value comes
from the same registry (``evaluation.problems.PROBLEM_REGISTRY`` / ``evaluation.objects.NAME_TO_OBJECT``) through
the same steps, for every problem at once; the sub-goal sequences are padded to the longest one (screwing: 10).

Only the goal modes without a per-episode prelude are supported ("preInsertAndFinal", the official default, and
"finalGoalOnly"); the prelude of "transportPreInsertFinal" is built from the start pose and is not used by the
released problems' default config.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

RELEASED_PROBLEMS = ("tight_insertion", "beam_assembly_step1", "beam_assembly_step2", "screwing")
SUPPORTED_GOAL_MODES = ("preInsertAndFinal", "finalGoalOnly")


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    object_name: str
    object_urdf: str  # absolute
    receptive_urdf: str  # absolute
    object_scale: tuple  # grasp-box scale handed to the policy (observation "object_scales")
    hole_z_offset: float
    insert_poses: tuple  # sub-goal poses in the receptive frame, (x, y, z, qx, qy, qz, qw) each

    @property
    def num_goals(self) -> int:
        return len(self.insert_poses)


def _resolve_asset_path(repo_root: Path, path) -> Path:
    """Official ``precise_assembly_env._resolve_asset_path``."""
    p = Path(path)
    if p.is_absolute():
        return p
    for c in (repo_root / p, repo_root / "assets" / p):
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve asset path {path!r} under {repo_root}")


def load_problem_specs(names, goal_mode: str, repo_root: Path) -> list[ProblemSpec]:
    """The official ``_configure_problem`` for each problem in ``names``."""
    if goal_mode not in SUPPORTED_GOAL_MODES:
        raise NotImplementedError(f"goal_mode {goal_mode!r}: only {SUPPORTED_GOAL_MODES} are supported")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from evaluation.objects import NAME_TO_OBJECT
    from evaluation.problems import PROBLEM_REGISTRY

    specs = []
    for name in names:
        if name not in PROBLEM_REGISTRY:
            raise KeyError(f"unknown problem {name!r}; known: {sorted(PROBLEM_REGISTRY)}")
        prob = PROBLEM_REGISTRY[name]
        obj = NAME_TO_OBJECT[prob.insertion_object_name]
        poses = prob.insert_pose_rel_receptive
        if goal_mode == "finalGoalOnly":
            poses = (prob.final_insert_pose_rel_receptive,)
        specs.append(ProblemSpec(
            name=name,
            object_name=prob.insertion_object_name,
            object_urdf=str(_resolve_asset_path(repo_root, obj.urdf_path)),
            receptive_urdf=str(_resolve_asset_path(repo_root, prob.receptive_urdf)),
            object_scale=tuple(float(v) for v in obj.scale),
            hole_z_offset=float(prob.hole_z_offset),
            insert_poses=tuple(tuple(float(v) for v in p) for p in poses),
        ))
    return specs


class ProblemTable:
    """Device tensors of the per-task values, indexed by task id; ``per_env`` gathers them by each env's task."""

    def __init__(self, specs: list[ProblemSpec], device):
        self.specs = specs
        self.names = tuple(s.name for s in specs)
        n, g = len(specs), max(s.num_goals for s in specs)
        self.max_goals = g
        self.num_goals = torch.tensor([s.num_goals for s in specs], dtype=torch.long, device=device)
        self.object_scale = torch.tensor([s.object_scale for s in specs], dtype=torch.float32, device=device)
        self.hole_z_offset = torch.tensor([s.hole_z_offset for s in specs], dtype=torch.float32, device=device)
        # padded with each task's last pose (never read: a task's sub-goal index stays below its own count)
        pos = torch.zeros((n, g, 3), dtype=torch.float32, device=device)
        quat = torch.zeros((n, g, 4), dtype=torch.float32, device=device)
        for t, s in enumerate(specs):
            p = list(s.insert_poses) + [s.insert_poses[-1]] * (g - s.num_goals)
            p = torch.tensor(p, dtype=torch.float32, device=device)
            pos[t] = p[:, 0:3]
            quat[t] = torch.cat([p[:, 6:7], p[:, 3:6]], dim=-1)  # xyzw -> wxyz, as the official _xyzw_to_wxyz
        self.insert_pos_rel = pos
        self.insert_quat_wxyz = quat

    def summary(self) -> list[dict]:
        return [dict(task=s.name, object=s.object_name, goals=s.num_goals, object_scale=list(s.object_scale),
                     hole_z_offset=s.hole_z_offset, object_urdf=s.object_urdf, receptive_urdf=s.receptive_urdf)
                for s in self.specs]
