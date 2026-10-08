"""Config of the hetero (union-scene) Factory env for Isaac Lab / PhysX.

Union scene: every env holds the parts of all three official Factory tasks (8 part slots: peg + hole,
medium gear + gear base + small and large gears, nut + bolt). Each env runs one task at a time; the parts of
the other two tasks sit at a parking spot behind the robot, out of reach. Every slot is its own Isaac Lab
``Articulation`` with one instance per env, so all envs have the same structure and
``replicate_physics`` can stay on.

Everything physical is the official config, unchanged:

* each slot's ``ArticulationCfg`` is the official one from ``factory_tasks_cfg`` with only ``prim_path``
  renamed (two tasks cannot both own ``/FixedAsset``);
* robot, controller (``ctrl``), simulation (``sim``) and observation noise (``obs_rand``) are inherited from
  the official ``FactoryEnvCfg``;
* per-task parameters live in ``task_table.py``, read from the official task configs.

What is new, and why:

* ``task_onehot`` is appended to the policy and critic inputs (RULES.md: shared policy with a task id).
* ``parking_pos``: where idle parts wait. The official action bound keeps the fingertip within 5 cm of the
  active fixed asset (``ctrl.pos_action_bounds``), and the reset IK targets a point above it, so parts
  0.6 m behind the robot are never touched. Slots are 0.3 m apart, so their bounding boxes never overlap
  and PhysX generates no contacts between them. One exception: the small and large gears have gravity, so
  when parked they rest on the parked gear base's shafts, as they do on the active base. Every non-gear env
  therefore carries those resting contacts; they are part of the union overhead to measure.
* ``episode_length_s``: the official tasks last 10 / 20 / 30 s, but the official reset (gravity off for the
  whole scene, then about 60 physics steps of IK and grasping) only works when every env resets at once.
  How to reconcile the two is an open team question; for now every env uses the longest (30 s).
* ``assignment``: hetero or homo, see ``assignment.py``.

Isaac Lab must be importable (launch the app first). Used for all PhysX runs on gputw1 (Isaac Sim 5.1, Isaac Lab v2.3.2).
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import GearMesh, NutThread, PegInsert

from .task_table import TASK_NAMES

HETERO_OBS_DIM_CFG = {**OBS_DIM_CFG, "task_onehot": len(TASK_NAMES)}
HETERO_STATE_DIM_CFG = {**STATE_DIM_CFG, "task_onehot": len(TASK_NAMES)}

# slot name -> (task, role, attribute of the official task config holding its ArticulationCfg, prim name)
PART_SLOTS = {
    "peg_fixed": ("peg_insert", "fixed", "fixed_asset", "PegInsert_FixedAsset"),
    "peg_held": ("peg_insert", "held", "held_asset", "PegInsert_HeldAsset"),
    "gear_fixed": ("gear_mesh", "fixed", "fixed_asset", "GearMesh_FixedAsset"),
    "gear_held": ("gear_mesh", "held", "held_asset", "GearMesh_HeldAsset"),
    "gear_small": ("gear_mesh", "small_gear", "small_gear_cfg", "GearMesh_SmallGearAsset"),
    "gear_large": ("gear_mesh", "large_gear", "large_gear_cfg", "GearMesh_LargeGearAsset"),
    "nut_fixed": ("nut_thread", "fixed", "fixed_asset", "NutThread_FixedAsset"),
    "nut_held": ("nut_thread", "held", "held_asset", "NutThread_HeldAsset"),
}

# task -> role -> slot
TASK_SLOTS: dict[str, dict[str, str]] = {}
for _slot, (_task, _role, _attr, _prim) in PART_SLOTS.items():
    TASK_SLOTS.setdefault(_task, {})[_role] = _slot

# Parking spots, metres in the env frame (robot base at the origin, table and active parts at x ~ +0.6).
# Behind the robot, 0.3 m apart along y, inside the env's 2 m cell. Held parts have gravity disabled in the
# official config, so they float where they are put; fixed parts are welded to the world. The small and
# large gears have gravity and are parked at the parked gear base's pose, which seats them on its shafts
# (the official reset places them the same way on the active base).
PARKING_POS = {
    "peg_fixed": (-0.6, -0.75, 0.05),
    "peg_held": (-0.6, -0.45, 0.10),
    "gear_fixed": (-0.6, -0.15, 0.05),
    "gear_held": (-0.6, 0.15, 0.10),
    "gear_small": (-0.6, -0.15, 0.05),  # on the parked gear base
    "gear_large": (-0.6, -0.15, 0.05),  # on the parked gear base
    "nut_fixed": (-0.6, 0.45, 0.05),
    "nut_held": (-0.6, 0.75, 0.10),
}

_OFFICIAL_TASKS = {"peg_insert": PegInsert(), "gear_mesh": GearMesh(), "nut_thread": NutThread()}
_OFFICIAL_ENV = FactoryEnvCfg()


def official_slot_cfg(slot: str) -> ArticulationCfg:
    """The official ArticulationCfg of a part, with only its prim path renamed."""
    task, _role, attr, prim = PART_SLOTS[slot]
    return getattr(_OFFICIAL_TASKS[task], attr).replace(prim_path=f"/World/envs/env_.*/{prim}")


@configclass
class AssignmentCfg:
    mode: str = "hetero"  # "hetero" | "homo"
    layout: str = "interleaved"  # hetero only: "interleaved" | "blocked" (see assignment.py)
    homo_task: int = 0  # homo only: the task of the first episodes
    # homo only, "episode" switching: move to the next task after this many complete episodes (0 = never; the
    # training script may instead switch at rollout boundaries with HeteroFactoryEnv.schedule_homo_switch)
    homo_switch_every_episodes: int = 0
    # hetero only, analysis experiments: fixed envs per task (e.g. (128, 43, 43)); None = the even split of RULES
    counts: tuple | None = None


@configclass
class HeteroFactoryEnvCfg(FactoryEnvCfg):
    """Official FactoryEnvCfg plus the union scene. ``task`` and ``task_name`` from the parent are unused."""

    task_names: tuple = TASK_NAMES

    obs_order: list = _OFFICIAL_ENV.obs_order + ["task_onehot"]
    state_order: list = _OFFICIAL_ENV.state_order + ["task_onehot"]

    # Open team question (see module docstring); "max" = every task uses the longest official episode.
    episode_length_mode: str = "max"
    episode_length_s = 30.0

    assignment: AssignmentCfg = AssignmentCfg()

    peg_fixed: ArticulationCfg = official_slot_cfg("peg_fixed")
    peg_held: ArticulationCfg = official_slot_cfg("peg_held")
    gear_fixed: ArticulationCfg = official_slot_cfg("gear_fixed")
    gear_held: ArticulationCfg = official_slot_cfg("gear_held")
    gear_small: ArticulationCfg = official_slot_cfg("gear_small")
    gear_large: ArticulationCfg = official_slot_cfg("gear_large")
    nut_fixed: ArticulationCfg = official_slot_cfg("nut_fixed")
    nut_held: ArticulationCfg = official_slot_cfg("nut_held")

    parking_pos: dict = PARKING_POS

    # Read each env's active part pose with one PhysX call per role over all tasks' slots (as the official env
    # makes one call per role). False: one call per task (the version used up to 2026-10-06; Hetero then makes
    # three times as many reads as Homo, about 2% of a step at 128 envs). Same values either way.
    fast_active_pose: bool = True


@configclass
class HeteroFactoryListEnvCfg(HeteroFactoryEnvCfg):
    """Big-list scene (hetero_factory_list_env.py): each env holds only its own task's parts, so the envs are
    not copies of each other and PhysX must parse every env (``replicate_physics=False``; ``clone_in_fabric``
    needs replicated envs). Everything else as the union scene. Hetero only: each env's task is fixed when the
    scene is built (``assignment.layout``), so leftover envs cannot rotate."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=2.0, replicate_physics=False, clone_in_fabric=False
    )
