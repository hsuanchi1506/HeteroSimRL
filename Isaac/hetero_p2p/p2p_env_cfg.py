"""Config of the union-scene Play2Perfect env (``HeteroP2PEnv``) for Isaac Lab / PhysX.

The official ``PreciseAssemblyEnvCfg`` (plus the official task YAML ``cfg/task/PreciseAssembly.yaml``, applied by
the training script) with:

* ``problems``: the tasks in the scene, the four released ones by default. ``precise_assembly.problem`` is unused.
* ``assignment``: hetero (env i runs task i % 4 for the whole run) or homo (every env runs ``homo_task``; the
  training script switches tasks at rollout boundaries).
* parking spots for the parts of the tasks an env is not running:

  - objects are dynamic and have gravity. They hang at ``object_park_z`` metres (the arm on its 0 m base reaches
    about 1.5 m) above their own spot, held by an upward force equal to their weight (``active_views.py``), so
    they touch nothing. Spots are 0.6 m apart in x / y and 0.5 m apart in z;
  - receptive parts (holes) and goal markers are kinematic; they wait below the table at ``hole_park_z`` /
    ``goal_park_z``, as the official env hides the hole of its random-goal envs (``hole_pos z = -1``).

Isaac Lab and the Play2Perfect repository (``isaacsimenvs``) must be importable: launch the app first.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg import PreciseAssemblyEnvCfg

from .problem_table import RELEASED_PROBLEMS


@configclass
class P2PAssignmentCfg:
    mode: str = "hetero"  # "hetero" | "homo"
    homo_task: int = 0  # homo: the task of the first episodes


@configclass
class HeteroP2PEnvCfg(PreciseAssemblyEnvCfg):
    problems: tuple = RELEASED_PROBLEMS
    assignment: P2PAssignmentCfg = P2PAssignmentCfg()

    # parking spots in the env frame (robot base at (0, 0.8, 0), table and active parts near the origin)
    object_park_xy: tuple = ((-0.3, -0.3), (-0.3, 0.3), (0.3, -0.3), (0.3, 0.3))
    object_park_z: float = 2.5  # task t hangs at object_park_z + 0.5 * t
    hole_park_z: float = -1.0
    goal_park_z: float = -1.0

    # measurement only (scripts/bench_p2p.py): give the objects PhysX's contact-report API so that contacts can
    # be read per env. Off in training, as in the official env.
    object_contact_sensors: bool = False
