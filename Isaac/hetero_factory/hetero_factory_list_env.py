"""HeteroFactoryListEnv: the "big list" version of the hetero Factory env (PhysX plan, option C).

Each env holds only the parts of its own task: no parked parts, no padding. All envs' bodies still live in one
PhysX scene, so the scene is one big list of every env's own parts; the union scene (HeteroFactoryEnv) instead
gives every env every part. Comparing the two on the same GPU measures what the padding costs in PhysX.

How it differs from the union scene, and why:

* Envs are built independently (``replicate_physics=False`` in the cfg). The official FactoryEnv builds env 0
  and then calls ``scene.clone_environments``, which makes every env inherit env 0's prims: that would put env
  0's parts into every env. Here ``clone_environments`` is not called; with ``replicate_physics=False`` the
  InteractiveScene already created every env prim at its grid position.
* Each part slot is spawned only in its task's envs: the spawner's clone decorator takes a regex for the parent
  prim (``/World/envs/env_(0|3|6)/...``). The Articulation is then created without spawning, with the usual
  ``env_.*`` pattern; PhysX turns ``.*`` into a glob, which matches only the envs that hold the prim. So a slot's
  Articulation has one row per env of its task, in the view's order (read from the view's prim paths).
* Every access to a slot goes through ``_slot_rows`` (global env id -> row of the slot's Articulation).
* ``scene.reset(env_ids)`` would index every Articulation with global env ids, out of range for the slots; all
  envs reset together anyway (as in the official env), so the base reset is called with ``None`` (= all).
* Each env's task is fixed when the scene is built (``assignment.layout``). RULES.md asks leftover envs to
  rotate across tasks; that needs every env to own every task's parts, i.e. the union scene. With 128 envs the
  split stays 43 / 43 / 42 for the whole run (``samples_per_task`` in metrics.jsonl shows it).
* Hetero only. Homo in the big-list layout is just the official single-task FactoryEnv.

Not run yet: written 2026-10-04 before access to the GPU machine, checked offline with a fake runtime.
"""

from __future__ import annotations

import re

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .assignment import hetero_assignment
from .hetero_factory_env import HeteroFactoryEnv
from .hetero_factory_env_cfg import PART_SLOTS, HeteroFactoryListEnvCfg


class HeteroFactoryListEnv(HeteroFactoryEnv):
    cfg: HeteroFactoryListEnvCfg

    def __init__(self, cfg: HeteroFactoryListEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.assignment.mode != "hetero":
            raise ValueError("the big-list scene fixes each env's task when it is built: hetero only")
        if cfg.scene.replicate_physics:
            raise ValueError("the big-list scene needs scene.replicate_physics=False (envs differ)")
        # each env's task, fixed before the scene is built
        self._fixed_task_ids = hetero_assignment(
            cfg.scene.num_envs, len(cfg.task_names), 0, cfg.assignment.layout, "cpu"
        )
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        """Robot and table in every env; each part slot only in the envs of its task."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._slots = {}
        self._slot_env_ids = {}
        task_names = tuple(self.cfg.task_names)
        for slot, (task, _role, _attr, prim) in PART_SLOTS.items():
            envs = (self._fixed_task_ids == task_names.index(task)).nonzero(as_tuple=False).squeeze(-1).tolist()
            if not envs:
                continue
            slot_cfg = getattr(self.cfg, slot)
            alternatives = "|".join(str(i) for i in envs)
            slot_cfg.spawn.func(
                f"/World/envs/env_({alternatives})/{prim}",
                slot_cfg.spawn,
                translation=slot_cfg.init_state.pos,
                orientation=slot_cfg.init_state.rot,
            )
            self._slots[slot] = Articulation(slot_cfg.replace(spawn=None))
            self._slot_env_ids[slot] = envs

        self._setup_extra_scene()  # e.g. the video script's cameras (spawned in every env: no cloning here)
        # no clone_environments(): it would make every env inherit env 0's parts (see the module docstring)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        for slot, asset in self._slots.items():
            self.scene.articulations[slot] = asset

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_tensors(self):
        # global env id -> row of each slot's Articulation, in the order of the PhysX view
        self._rows = {}
        self._counts = {}
        for slot, asset in self._slots.items():
            envs = self._view_env_ids(asset)
            expected = sorted(self._slot_env_ids[slot])
            if sorted(envs) != expected:
                raise RuntimeError(f"slot {slot}: view holds envs {sorted(envs)[:8]}..., expected {expected[:8]}...")
            rows = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            rows[torch.tensor(envs, device=self.device)] = torch.arange(len(envs), device=self.device)
            self._rows[slot] = rows
            self._counts[slot] = len(envs)
        super()._init_tensors()

    def _view_env_ids(self, asset) -> list[int]:
        """Env index of each row of the asset's PhysX view, from the view's prim paths."""
        paths = getattr(asset.root_physx_view, "prim_paths", None)
        if paths is None:  # older PhysX tensor API: rows follow creation order, i.e. ascending env index
            return sorted(self._slot_env_ids[[k for k, v in self._slots.items() if v is asset][0]])
        return [int(re.search(r"/env_(\d+)(/|$)", p).group(1)) for p in paths]

    # ------------------------------------------------------------------ slot access

    def _slot_rows(self, slot: str, env_ids: torch.Tensor) -> torch.Tensor:
        return self._rows[slot][env_ids]

    def _slot_count(self, slot: str) -> int:
        return self._counts[slot]

    def _park(self, slot: str, env_ids: torch.Tensor):
        """Nothing to park: a slot exists only in the envs of its task."""

    def _next_task_ids(self) -> torch.Tensor:
        return self._fixed_task_ids.to(self.device)

    def _base_reset_idx(self, env_ids):
        # scene.reset(env_ids) indexes every Articulation with global env ids, out of range for the part slots;
        # every env resets here anyway, and None means all
        DirectRLEnv._reset_idx(self, None)
