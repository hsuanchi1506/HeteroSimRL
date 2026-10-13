# Isaac / PhysX

Homo and Hetero versions of the official Isaac Lab Factory tasks (PegInsert, GearMesh, NutThread) and of the four Play2Perfect assembly tasks, following `../RULES.md`. Chinese version: [README.zh-TW.md](README.zh-TW.md). Results: [results/README.md](results/README.md).

## Status (2026-10-10)

- **Done**:
  - The union-scene environment, shared by Homo and Hetero.
  - Training, evaluation, gradient diagnostics, video, timing and profiling tools.
  - Hetero/Homo training at 128 envs: official Hetero with 3 seeds, the other two groups with 5 seeds each.
  - RULES section 6 with trained policies at 3 / 128 / 4096 envs: contacts, contact phases and waiting.
  - Section 7: reading the PhysX source.
  - The TODOs from the 2026-10-07 meeting: advantage normalisation for Homo, splitting the simulation over processes, finer profiling.
  - The RULES "run once" set (`rules_*`) at 128, 3 and 4096 envs, with periodic evaluation and videos.
- **Play2Perfect** (2026-10-13): Homo/Hetero trained (one seed). Hetero finished; Homo runs out of GPU memory at the official scale (kept as a result). See [Play2Perfect](#play2perfect-in-progress) and [results/README.md](results/README.md).

## Setup

| | |
|---|---|
| Machine | gputw1: RTX PRO 6000 Blackwell Workstation (96 GB), driver 595.84, 500 W power cap, shared |
| Software | Isaac Sim 5.1.0, Isaac Lab v2.3.2 (source), PyTorch 2.7.0+cu128, rl_games 1.6.1 (the version Isaac Lab installs), Python 3.11 (conda) |
| Running | From the IsaacLab folder, with HeteroSimRL next to it: `./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/<script>.py --headless ...` |
| PhysX settings (official) | physics dt 1/120 s; 8 physics steps per policy step; 192 position iterations; official GPU buffers except the collision stack at >= 1024 envs (see below) |

Note: on gputw1 the 595 driver makes Isaac Sim 5.1's RTX renderer crash at start-up, even with an empty scene. Headless simulation and training are unaffected; videos are made by recording poses in PhysX and drawing them with MuJoCo (see "Videos").

## Approach: union scene

The official Factory env runs one task. For Hetero, the same batch of envs must run all three tasks:

- **Parts**: every env holds all 8 parts: peg and hole, medium gear and gear base, small and large gear, nut and bolt. An env runs one task at a time; the other tasks' parts are parked 0.6 m behind the robot, out of its reach.
- **Part configs**: every part is the official `ArticulationCfg` with only its prim path renamed, since two tasks cannot both own `/FixedAsset`.
- **Inherited settings**: the robot, controller, simulation settings and observation noise are inherited from the official `FactoryEnvCfg`.
- **Task table**: the per-task parameters (initial states, noise, reward coefficients, success thresholds, ...) are read from the official configs and functions into a table, never copied by hand (`TASK_PARAMS.md`). Each env indexes the table by its task id.
- **Observations**: they gain a task one-hot (policy 22 dims, critic 46 dims).
- **Homo and Hetero** use the same scene; they differ only in which task each env gets.

Checked so far:

- In Homo mode with the same seed, poses, observations and rewards match the official `FactoryEnv` for all three tasks (offline check).
- The union scene and the big-list scene (each env holds only its own parts) give identical poses at every step, so the parked parts do not change the physics.

`MultiUsdFileCfg` was not used because it needs identical prim names in every file, and the official parts' names differ.

## Files

| File | Contents |
|---|---|
| `hetero_factory/hetero_factory_env_cfg.py` | Union scene: 8 part slots, parking poses, `task_onehot`, task assignment settings |
| `hetero_factory/hetero_factory_env.py` | The env: subclasses the official `FactoryEnv` and overrides only task-dependent methods, each change marked `HETERO`. Gym id `Isaac-HeteroFactory-Direct-v0` |
| `hetero_factory/hetero_factory_list_env.py` | Big-list scene (each env holds only its task's parts). Gym id `Isaac-HeteroFactoryList-Direct-v0` |
| `hetero_factory/task_table.py`, `TASK_PARAMS.md`, `print_task_table.py` | Per-task parameter table, its contents and sources, and a script to regenerate it on the training machine |
| `hetero_factory/assignment.py` | Which task each env runs: Hetero even split (leftover envs rotate every episode), Homo all on one task; fixed counts for analysis runs |
| `hetero_factory/train_common.py`, `train_common_args.py` | Shared by both training scripts: official PPO config, the changes RULES allows, logging |
| `hetero_factory/rl_utils.py` | `env.step` timing, GPU sampling, one log line per epoch, per-task / pooled advantage normalisation, Homo switching |
| `hetero_factory/grad_probe.py` | Gradient diagnostics (RULES section 5) |
| `hetero_factory/profiling.py` | Groups CUDA kernels (contact detection, solver, articulation, ...) |
| `hetero_factory/pose_record.py`, `video_utils.py` | Video helpers |
| `hetero_factory/offline_check/` | Checks that run without Isaac Lab (stub modules); see its README |
| `scripts/train_hetero.py`, `train_homo.py` | Training, adapted from Isaac Lab's official rl_games `train.py` |
| `scripts/eval_policy.py` | Evaluation: each task separately, deterministic actions |
| `scripts/analyze_runs.py`, `plot_groups.py` | Success and return vs env steps and wall-clock; groups of seeds as mean and range; gradient figures |
| `scripts/record_video.py`, `render_offline.py` | Videos: poses from PhysX, drawn with MuJoCo |
| `scripts/bench_scene.py` | Random-action timing: union vs big list, per-stage times, PhysX statistics |
| `scripts/bench_policy.py` | Trained-policy timing (RULES section 6): whole episodes, contact points and phases, profiling windows, fixed task counts, synchronised start of several processes |
| `scripts/profile_step.py` | Kernel profile with the PyTorch profiler, or a capture range for Nsight Systems |
| `scripts/check_active_pose.py` | Checks that the one-read pose of the active parts equals the per-task read |
| `scripts/smoke_env.py` | First smoke test on a new machine |

## Running

```bash
# training (official PPO, 200 epochs)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_hetero.py --num_envs 128 --seed 0 --headless
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/train_homo.py --num_envs 128 --seed 0 --switch rollout --block 4 --headless
# evaluation (an nn/ directory picks its last checkpoint)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/eval_policy.py --headless \
    --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn --out runs/eval/hetero_n128_s0.json
# figures (no Isaac Sim needed)
python ../HeteroSimRL/Isaac/scripts/plot_groups.py --out runs/analysis \
    --group "Hetero" tab:blue logs/rl_games/Factory/hetero_n128_s0 --group "Homo" tab:orange logs/rl_games/Factory/homo_rollout4_n128_s0
# trained-policy timing (section 6)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/bench_policy.py --headless --num_envs 128 \
    --checkpoint logs/rl_games/Factory/hetero_n128_s1_ptadv/nn --contacts --profile_windows 5,300 --out runs/sec6/n128_hetero.json
```

Common training options:

- `--num_envs`: 3, 128 or 4096.
- `--seed`.
- `--adv_norm global|per_task|pooled`: advantage normalisation; see below.
- `--save_every N`: checkpoint every N epochs, for periodic evaluation and videos. Official 100; this changes only the checkpoint frequency, not PPO.
- `--collision_stack_size`: PhysX collision stack in bytes.
- `--grad_every`: how often to run the gradient diagnostics.

Output in `logs/rl_games/Factory/<run>/`:

- `run_info.json`: settings, changes to the official config, versions.
- `metrics.jsonl`: one line per epoch: times, `env.step` time, GPU, samples per task, advantage statistics.
- `episodes.jsonl`: one line per episode and task.
- `grad_diag.jsonl`: gradient diagnostics.

## Changes to the official settings

Required or allowed by RULES:

| Change | Why |
|---|---|
| Minibatch 512 → 384 at 3 envs | A rollout holds only 3 × 128 = 384 samples (RULES allows changes for divisibility) |
| Larger collision stack at >= 1024 envs (Hetero 1 GB, Homo 2 GB) | The official 2^28 bytes fit 128 envs only; beyond that PhysX drops contacts. A simulator buffer, not a task or PPO setting |
| Every episode 30 s | Official episodes are 10 / 20 / 30 s (peg / gear / nut), but the official reset resets all envs together; the longest is used until the team decides |
| `save_frequency` 20 (RULES runs only) | Section 8 asks for periodic evaluation and videos |

Extra analysis, each behind a switch; off means the official setting:

| Item | Why |
|---|---|
| `--adv_norm per_task` | The official setting normalises the advantages of the whole rollout together. A Hetero rollout mixes three tasks, so the nut task, with the smallest rewards, is pushed negative as a whole and some seeds never learn it. Normalising each task separately fixes it: this run shows the cause |
| `--adv_norm pooled` | From the 2026-10-07 meeting. A Homo rollout holds one task, so the official normalisation is already per task for Homo. This option normalises with statistics pooled over the latest rollout of every task: the Homo counterpart of what the official setting does in Hetero, and identical to the official setting in Hetero |
| One-read pose of the active parts (`fast_active_pose`, on by default) | The old code read the parts once per task every physics step, three times in Hetero: about 2% of a step. One read gives identical values (`check_active_pose.py`) |
| `bench_policy.py`, `profile_step.py`, Nsight | Section 6 asks for per-task step time, contacts and waiting. Random actions never mate the parts and underestimate the cost of mixing, so trained policies are used |
| `--task_counts` (timing only) | From the 2026-10-07 meeting: uneven envs per task, to fill the waiting time or group tasks by difficulty |

## Big-list scene (`--scene list`)

Each env holds only its task's parts; all envs are still in one PhysX scene.

- It measures what the parked parts of the union scene cost:
  - 2% at 3 and 128 envs;
  - about 1.8× slower union at 1024 and 4096 envs, because the parked gears sit on the parked base and their contacts are solved every step.
- Each env is built on its own (`replicate_physics=False`), because cloning env 0 would put its parts into every env.
- Each part is spawned only in its task's envs; all reads and writes go through a global-env-id → row table.
- Hetero only; each env's task is fixed when the scene is built.

## Advantage normalisation

rl_games normalises the advantages of the whole rollout to mean 0 and std 1.

- `per_task`: after rl_games builds the dataset and before the update (`after_steps`), the samples are split by the task one-hot in the observation and each task is normalised separately.
- `pooled`: the advantages from before rl_games normalises them are kept; each task's latest (count, mean, variance) is stored, and the advantages are normalised with the pooled mean and std of all tasks.
- Neither changes rl_games itself. `metrics.jsonl` logs each task's `adv_mean_<task>` and `adv_std_<task>` in every mode.

## Homo task switching

rl_games updates every 128 steps and an episode is 449 steps, so "one task per update" and "no cut episodes" cannot both hold.

- `--switch rollout` (default, RULES as written): the task changes every `--block` updates. All envs time out on the last step; the cut episodes are bootstrapped with the critic and logged as `_truncated`.
- `--switch episode`: the task changes every `--block` complete episodes; the update that spans a switch mixes two tasks.

Either way, after one cycle (a block of each task) Homo and Hetero have the same samples per task (`samples_per_task` in `metrics.jsonl`).

## Gradient diagnostics (RULES section 5)

Measured at the start of a PPO update, with the parameters before the update:

- per-task actor gradient norm;
- task-pair gradient cosine;
- cosine of each task to the total;
- update-to-update cosine: at epochs 1 and 2, then every `--grad_every` epochs on a pair of consecutive updates;
- the actual parameter change of every epoch.

Observation only: it does not touch `.grad`, the optimiser, the data or the global RNG. Its time is logged separately; `wall_time_excl_diag` is the wall time without it.

## Videos

```bash
# 1. run the physics and record every rigid body's pose (no cameras)
./isaaclab.sh -p ../HeteroSimRL/Isaac/scripts/record_video.py --headless \
    --checkpoint logs/rl_games/Factory/hetero_n128_s0/nn --seconds 30 --out runs/video/hetero_n128_s0
# 2. draw it with MuJoCo (a small separate env: numpy, mujoco, imageio-ffmpeg, pillow)
MUJOCO_GL=egl python ../HeteroSimRL/Isaac/scripts/render_offline.py runs/video/hetero_n128_s0.npz
```

- One row per task, 3 tiles per row (240 × 180 each); a tile turns green on success.
- On a machine with a working driver, `--renderer rtx --enable_cameras` renders in Isaac Sim directly.
- Videos are never recorded during timing or profiling.

## Play2Perfect (in progress)

Homo and Hetero versions of the four Play2Perfect assembly tasks (tight insertion, beam assembly steps 1 and 2, screwing). Code in `hetero_p2p/`. It runs in Play2Perfect's own environment (Isaac Lab 2.3.2.post1, the repository's rl_games with SAPG), from the Play2Perfect folder.

**The official stage 2.** Each task is finetuned on its own, starting from the released stage-1 ("play") policy. The YAML files leave the size open, but the released stage-2 checkpoints fix it: all four used 12,288 envs in 6 SAPG blocks of 2,048 (their saved rollouts hold 12,288 envs, whose block column takes 6 values on 2,048 envs each), and count epochs from 0, i.e. the play weights were loaded without resuming its counters. Those are the defaults here. The YAML's block size (4,096) gives only 3 blocks, and the play policy, which has 6, does not load.

**Union scene**, as for Factory:

- Every env holds the four tasks' objects, receptive parts (holes) and goal markers.
- Holes and markers are kinematic. When parked they wait below the table (z = -1 m), as the official env hides the hole of its random-goal envs.
- Objects are dynamic. When parked they hang 2.5-4 m up, held by an upward force equal to their weight, so they touch nothing. Their pose is rewritten at every reset, so float rounding cannot make them drift.
- `ActiveRigidObject` stands in for the official `env.object` / `env.hole` / `env.goal_viz`. Reads return each env's task's part; writes go to that part only. So the official observation, reward, reset and termination code runs unchanged.
- Per-env values come from the official problem registry, never copied by hand: object scale (an observation), hole height, sub-goal poses (padded to 10, the screwing count), sub-goal count, object mass.

**Policy input**: a task one-hot (4 values) is appended to the policy and critic inputs, before SAPG's block column. The released play checkpoint is widened with zero weights for it, so its outputs are unchanged (checked: identical).

**Task assignment**:

- Hetero: env i runs task i % 4 for the whole run. Envs reset one by one in this env, so no rotation is needed, and every SAPG block holds every task equally.
- Homo: every env runs one task. The task changes every 40 epochs (640 steps, about one episode) at a rollout boundary; the cut episodes are counted apart.

| File | Contents |
|---|---|
| `hetero_p2p/problem_table.py` | The four problems' values from the official registry, as per-task tensors |
| `hetero_p2p/active_views.py` | `ActiveRigidObject`: the active part per env; parking; anti-gravity force for parked objects |
| `hetero_p2p/p2p_env_cfg.py`, `p2p_env.py` | The union-scene env: subclasses the official `PreciseAssemblyEnv`; changes marked `HETERO` |
| `hetero_p2p/checkpoint_utils.py` | Widens a released checkpoint for the one-hot (zero weights) |
| `hetero_p2p/train_utils.py` | Per-epoch metrics (episodes per task and SAPG block), light checkpoints, Homo switching |
| `scripts/train_p2p.py` | Training, adapted from Play2Perfect's `isaacsimenvs/train.py` |
| `scripts/eval_p2p.py` | Per-task success with the official offline method (first episode per env, deterministic, randomisation off) |
| `scripts/check_p2p_env.py` | Checks: parking, active parts, Homo switch |
| `scripts/plot_p2p.py` | Success and return per task vs env steps and wall-clock (all envs or SAPG's last block) |
| `hetero_p2p/pose_record_p2p.py`, `scripts/record_video_p2p.py` | Videos as for Factory: poses from PhysX (parked parts hidden), drawn by `render_offline.py --ground_z 0`. The mesh export now also reads USD boxes, cylinders and spheres (Play2Perfect's table, L-peg and hole are boxes) |

```bash
cd ~/HeteroSimRL/Isaac && python -m hetero_p2p.checkpoint_utils     ~/play2perfect/pretrained_policy/model.pth ~/runs/p2p_ckpt/play_onehot4.pth --n_new 4
cd ~/play2perfect
python ~/HeteroSimRL/Isaac/scripts/train_p2p.py --headless --mode hetero --epochs <N>     --checkpoint ~/runs/p2p_ckpt/play_onehot4.pth --save_every 250
python ~/HeteroSimRL/Isaac/scripts/train_p2p.py --headless --mode homo --block 40 --epochs <N>     --checkpoint ~/runs/p2p_ckpt/play_onehot4.pth --save_every 250
python ~/HeteroSimRL/Isaac/scripts/eval_p2p.py --headless --checkpoint <run>/light/ep<N>.pth --out <file>.json
```

Checked so far (2026-10-09, CPU physics, 24 envs; the GPU was busy with timed Factory runs):

- The widened play policy gives exactly the released one's actions, action noise, values and LSTM states.
- The scene builds; inputs are 144 (policy) and 166 (critic) values; the one-hot matches each env's task; sub-goal counts are 2 / 2 / 2 / 10.
- Parked objects stay within 0.1 µm of their spots over 60 steps.
- A Homo switch moves every env to the next task and updates the one-hot, sub-goal count and object scale; Homo rollouts hold one task.
- Training runs and writes metrics and checkpoints.
- Not checkable on CPU: PhysX's CPU pipeline does not keep a pose written to a kinematic body (the table and holes jump back), so where the active parts end up must be checked on GPU.

Checked on GPU (2026-10-10):

- Parked parts stay put; active holes sit on the table (z about 0.53 m) and active objects rest on the table or hole; the Homo switch works.
- **The four released assembly policies, widened for the one-hot and run in the union scene with the official offline method (1,024 envs per task), reproduce the official numbers**:

  | Task | Union scene | Official (`eval_offline.py`, same machine) |
  |---|---|---|
  | Tight insertion | 95.9% | 96.9% |
  | Beam assembly, step 1 | 97.3% | 97.1% |
  | Beam assembly, step 2 | 92.3% | 92.2% |
  | Screwing | 61.0% | 61.4% |

  Each released policy succeeds only on its own task (0-25% on the others).
- Videos of two released policies: tables, fixtures and parts in place; pegs inserted, legs screwed in.
- Speed at 12,288 envs (simulation time of one 16-step rollout): Homo tight insertion about 1.0 s, beam step 1 about 2.2 s, beam step 2 about 1.5 s, screwing 8.5-11.6 s; Hetero about 9 s. **The mixed batch is as slow as its slowest task (screwing)**, as Factory at small scale, but here that is about 2.4 times the Homo mean (Factory: +11%). Scene build about 11 min; GPU memory up to about 48 GB.

Training (2026-10-10 to 10-13; results in [results/README.md](results/README.md)): Hetero and Homo, 10,000 epochs each (2.0e9 env steps, about 2,500 single-task epochs per task; the released per-task finetunes ran 7,000-14,000 epochs and reached their success within 2-5 h, about 1,100-2,700 epochs), then the official offline evaluation every 2,000 epochs and videos.

Changes to the official stage 2: the task one-hot (and the widened checkpoint); the union scene and parking; a fixed number of epochs (the official config runs until stopped).

## Open team questions

- **Episode length**: 30 s for every task, or each task's official length?
- **Homo switching**: how and when the task changes.
- **Per-task advantage normalisation for Hetero**: should it be used? The official setting penalises Hetero.
- **Timing**: use trained policies. Random actions underestimate the cost of mixing (6-9% with random actions vs 12% during training).
- **From the 2026-10-07 meeting**: more tasks (e.g. 128), off-policy RL, reward changes. All of these go beyond RULES.
